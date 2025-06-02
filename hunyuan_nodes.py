import torch
import numpy as np
from PIL import Image
import os
import imageio
import decord
import math
from tqdm import tqdm

from comfy.model_management import get_torch_device
import comfy.utils
import folder_paths

from .hunyuan_utils import (
    load_hunyuan_models, load_onnx_utils, hunyuan_comfy_preprocess,
    load_embedded_yaml_config
)
from .vendored.dataset.utils import seed_everything
from .vendored.pipelines.hunyuan_svd_pipeline import HunyuanLongSVDPipeline

def comfy_image_to_pil(image_tensor):
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor[0]
    elif image_tensor.ndim != 3:
        raise ValueError(f"comfy_image_to_pil expects a 3D (H,W,C) or 4D (1,H,W,C) tensor, got {image_tensor.shape}")
    return Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))

def save_frames_to_temp_video_file(driving_video_input_data, user_specified_fps=25):
    video_frames_tensor_thwc = None
    actual_fps_to_use = user_specified_fps

    if isinstance(driving_video_input_data, torch.Tensor):
        if driving_video_input_data.ndim == 5 and driving_video_input_data.shape[0] == 1:
            video_frames_tensor_thwc = driving_video_input_data[0]
        elif driving_video_input_data.ndim == 4:
            video_frames_tensor_thwc = driving_video_input_data
        else:
            raise ValueError(f"save_frames: Input video tensor has unsupported ndim: {driving_video_input_data.ndim}. Expected 4 or 5.")
    elif isinstance(driving_video_input_data, tuple) and len(driving_video_input_data) > 0:
        if isinstance(driving_video_input_data[0], torch.Tensor) and driving_video_input_data[0].ndim == 4:
            video_frames_tensor_thwc = driving_video_input_data[0]
            try:
                if len(driving_video_input_data) > 2 and isinstance(driving_video_input_data[2], (int, float)) and driving_video_input_data[2] > 0:
                    actual_fps_to_use = int(round(driving_video_input_data[2]))
                elif len(driving_video_input_data) > 7 and isinstance(driving_video_input_data[7], (int, float)) and driving_video_input_data[7] > 0:
                    actual_fps_to_use = int(round(driving_video_input_data[7]))
            except (IndexError, TypeError): pass
        else:
            raise TypeError(f"save_frames: Input video is a tuple, but first element is not a 4D tensor. Got: {type(driving_video_input_data[0])}")
    else:
        raise TypeError(f"save_frames: Unsupported driving_video data type: {type(driving_video_input_data)}. Expected tensor or specific tuple.")

    if video_frames_tensor_thwc is None:
        raise ValueError("save_frames: Could not extract video frames tensor from input.")

    output_dir = folder_paths.get_temp_directory()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subfolder_name = f"hunyuan_vid_{os.urandom(6).hex()}"
    temp_sub_dir_path = os.path.join(output_dir, subfolder_name)
    os.makedirs(temp_sub_dir_path, exist_ok=True)

    temp_video_path = os.path.join(temp_sub_dir_path, "temp_driving_video.mp4")

    writer = None
    try:
        writer = imageio.get_writer(temp_video_path, fps=actual_fps_to_use, macro_block_size=1)
        num_frames_to_write = video_frames_tensor_thwc.shape[0]
        for i in range(num_frames_to_write):
            frame_np_uint8 = (video_frames_tensor_thwc[i].cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(frame_np_uint8)
    except Exception as e:
        if writer is not None: writer.close()
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if os.path.exists(temp_sub_dir_path) and not os.listdir(temp_sub_dir_path):
            os.rmdir(temp_sub_dir_path)
        raise RuntimeError(f"Error writing frames to temp video: {e}")
    finally:
        if writer is not None: writer.close()

    driving_data = {
        "video_path": temp_video_path,
        "temp_sub_dir": temp_sub_dir_path,
        "fps": actual_fps_to_use
    }
    return driving_data

class HunyuanPortrait_ModelLoader:
    vae_files = ["None"] + folder_paths.get_filename_list("vae")
    @classmethod
    def INPUT_TYPES(s):
        default_model_folder = os.path.join(folder_paths.models_dir, "HunyuanPortrait")
        default_model_folder_display = default_model_folder if os.path.isdir(default_model_folder) else "ComfyUI/models/HunyuanPortrait"
        return {
            "required": {
                "model_folder": ("STRING", { "default": default_model_folder_display }),
                "vae_name": (s.vae_files, {"default": "None"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }
    RETURN_TYPES = ("HY_MODELS",)
    RETURN_NAMES = ("hunyuan_models",)
    FUNCTION = "load_env"
    CATEGORY = "HunyuanPortrait"
    def load_env(self, model_folder, vae_name, precision, device):
        print("Loading Models...")

        abs_model_folder = model_folder
        if not os.path.isabs(model_folder):
            resolved_folder = os.path.join(folder_paths.base_path, model_folder)
            if os.path.isdir(resolved_folder): abs_model_folder = resolved_folder
        if not os.path.isdir(abs_model_folder):
            raise FileNotFoundError(f"HunyuanPortrait model folder not found: {abs_model_folder}")

        if vae_name == "None": raise ValueError("VAE name cannot be 'None'. Please select a VAE.")
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"Selected VAE file '{vae_name}' not found: {vae_path}")

        model_sub_paths = {
            "unet_path": ("UNet", os.path.join(abs_model_folder, "hyportrait", "unet.pth")),
            "image_encoder_path": ("Image Encoder", os.path.join(abs_model_folder, "hyportrait", "dino.pth")),
            "image_projector_path": ("Image Projector", os.path.join(abs_model_folder, "hyportrait", "image_proj.pth")),
            "pose_guider_path": ("Pose Guider", os.path.join(abs_model_folder, "hyportrait", "pose_guider.pth")),
            "motion_expression_path": ("Motion Expression", os.path.join(abs_model_folder, "hyportrait", "expression.pth")),
            "motion_headpose_path": ("Motion Headpose", os.path.join(abs_model_folder, "hyportrait", "headpose.pth")),
            "motion_refiner_path": ("Motion Refiner", os.path.join(abs_model_folder, "hyportrait", "motion_proj.pth")),
        }
        paths_for_loader = {}
        for name_key, (descriptive_name, path_val) in model_sub_paths.items():
            if not os.path.exists(path_val):
                raise FileNotFoundError(f"Hunyuan model part '{descriptive_name}' not found: {path_val}. Check '{abs_model_folder}' and its 'hyportrait' subdirectory.")
            paths_for_loader[name_key] = path_val

        device_str_used = device if device != "auto" else ('cuda' if get_torch_device().type == 'cuda' else 'cpu')

        hunyuan_models_dict = load_hunyuan_models(
            vae_path, paths_for_loader["unet_path"], paths_for_loader["image_encoder_path"],
            paths_for_loader["image_projector_path"], paths_for_loader["pose_guider_path"],
            paths_for_loader["motion_expression_path"], paths_for_loader["motion_headpose_path"],
            paths_for_loader["motion_refiner_path"],
            dtype_str=precision, device_str=device_str_used
        )
        hy_models_bundle = {
            "models": hunyuan_models_dict,
            "model_folder": abs_model_folder,
            "device": device_str_used
        }
        return (hy_models_bundle,)

class HunyuanPortrait_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        try:
            cfg = load_embedded_yaml_config()
            defaults = {"limit_frames": cfg.get("frame_num", 150), "limit_frames_max": 1000,
                        "image_size_preproc": cfg.get("arcface_img_size", 512),
                        "dwpose_area_scale": cfg.get("area", 1.25),
                        "use_arcface": cfg.get("use_arcface", True)}
        except Exception as e:
            print(f"[Hunyuan Preprocessor] Warning: Could not load embedded YAML config, using defaults. Error: {e}")
            defaults = {"limit_frames": 150, "limit_frames_max": 1000, "image_size_preproc": 512,
                        "dwpose_area_scale": 1.25, "use_arcface": True}
        return {
            "required": {
                "hunyuan_models": ("HY_MODELS",),
                "source_image": ("IMAGE",),
                "driving_video_frames": ("IMAGE",),
                "limit_frames": ("INT", {"default": defaults["limit_frames"], "min": 1, "max": defaults["limit_frames_max"]}),
                "output_crop_size": ("INT", {"default": defaults["image_size_preproc"], "min": 64, "max": 2048, "step": 8}),
                "dwpose_crop_area_scale": ("FLOAT", {"default": defaults["dwpose_area_scale"], "min": 0.5, "max": 3.0, "step": 0.05}),
                "use_arcface": ("BOOLEAN", {"default": defaults["use_arcface"]}),
                "driving_video_fps": ("INT", {"default": 25, "min":1, "max":60}),
            }
        }
    RETURN_TYPES = ("PREPROCESSED_DATA", "DRIVING_DATA")
    RETURN_NAMES = ("preprocessed_data", "driving_data")
    FUNCTION = "preprocess_data"
    CATEGORY = "HunyuanPortrait"

    def preprocess_data(self, hunyuan_models, source_image, driving_video_frames,
                        limit_frames, output_crop_size, dwpose_crop_area_scale, use_arcface, driving_video_fps):

        print("Preprocessor: Initializing...")

        model_folder = hunyuan_models["model_folder"]
        device_str_for_onnx = hunyuan_models["device"]

        onnx_arcface_path = os.path.join(model_folder, "arcface.onnx")
        onnx_yoloface_path = os.path.join(model_folder, "yoloface_v5m.pt")
        if not os.path.exists(onnx_arcface_path):
            raise FileNotFoundError(f"ArcFace ONNX model not found at: {onnx_arcface_path}")
        if not os.path.exists(onnx_yoloface_path):
            raise FileNotFoundError(f"YoloFace PT model not found at: {onnx_yoloface_path}")

        onnx_utils = load_onnx_utils(onnx_arcface_path, onnx_yoloface_path, device_str=device_str_for_onnx)
        source_image_pil = comfy_image_to_pil(source_image) # This line caused the error

        driving_data_dict = save_frames_to_temp_video_file(
            driving_video_frames, user_specified_fps=driving_video_fps
        )
        video_path_for_preprocessing = driving_data_dict["video_path"]
        temp_sub_dir_to_clean_later = driving_data_dict["temp_sub_dir"]

        actual_frames_to_process_for_pbar = limit_frames
        try:
            vr = decord.VideoReader(video_path_for_preprocessing, ctx=decord.cpu(0))
            actual_frames_in_video = len(vr)
            del vr
            actual_frames_to_process_for_pbar = min(actual_frames_in_video, limit_frames)
        except Exception as e_decord:
            print(f"Warning: Could not count frames in temp video for progress bar: {e_decord}. Using 'limit_frames' ({limit_frames}) as total.")

        pbar_comfy = comfy.utils.ProgressBar(actual_frames_to_process_for_pbar)
        tqdm_pbar = tqdm(total=actual_frames_to_process_for_pbar, desc="Preprocessing Frames")

        processed_frame_count_for_pbar = 0
        def callback_update_preprocessor_pbar():
            nonlocal processed_frame_count_for_pbar
            processed_frame_count_for_pbar += 1
            pbar_comfy.update(1)
            tqdm_pbar.update(1)

        pipeline_cfg = load_embedded_yaml_config()
        preprocessed_bundle = None
        try:
            preprocessed_bundle = hunyuan_comfy_preprocess(
                source_image_pil, video_path_for_preprocessing, onnx_utils, pipeline_cfg,
                limit_frames, output_crop_size, dwpose_crop_area_scale, use_arcface,
                progress_callback=callback_update_preprocessor_pbar
            )
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            try:
                if temp_sub_dir_to_clean_later and video_path_for_preprocessing and os.path.exists(video_path_for_preprocessing):
                    os.remove(video_path_for_preprocessing)
                if temp_sub_dir_to_clean_later and os.path.exists(temp_sub_dir_to_clean_later):
                    if not os.listdir(temp_sub_dir_to_clean_later): os.rmdir(temp_sub_dir_to_clean_later)
            except OSError as e_clean: print(f"Error cleaning up temp video/subdir after preprocess error: {e_clean}")
            raise
        finally:
            tqdm_pbar.close()

        if preprocessed_bundle is None:
             try:
                if temp_sub_dir_to_clean_later and video_path_for_preprocessing and os.path.exists(video_path_for_preprocessing):
                     os.remove(video_path_for_preprocessing)
                if temp_sub_dir_to_clean_later and os.path.exists(temp_sub_dir_to_clean_later) and not os.listdir(temp_sub_dir_to_clean_later):
                    os.rmdir(temp_sub_dir_to_clean_later)
             except OSError: pass
             raise ValueError("Preprocessing failed to return data.")
        print("Preprocessor: Finished.")
        return (preprocessed_bundle, driving_data_dict)

class HunyuanPortrait_Generator:
    @classmethod
    def INPUT_TYPES(s):
        try:
            cfg = load_embedded_yaml_config()
            defaults = { "height": cfg.get("height", 576), "width": cfg.get("width", 1024),
                         "num_inference_steps": cfg.get("num_inference_steps", 25),
                         "min_appearance_guidance_scale": cfg.get("min_appearance_guidance_scale", 1.0),
                         "max_appearance_guidance_scale": cfg.get("max_appearance_guidance_scale", 2.5),
                         "min_motion_guidance_scale": cfg.get("min_motion_guidance_scale", 1.0),
                         "max_motion_guidance_scale": cfg.get("max_motion_guidance_scale", 3.5),
                         "fps_generation": cfg.get("fps", 15), "motion_bucket_id": cfg.get("motion_bucket_id", 127),
                         "noise_aug_strength": cfg.get("noise_aug_strength", 0.02),
                         "decode_chunk_size": cfg.get("decode_chunk_size", 8),
                         "i2i_noise_strength": cfg.get("i2i_noise_strength", 1.0),
                         "n_sample_frames_batching": cfg.get("n_sample_frames", 14),
                         "pad_motion_frames": cfg.get("pad_frames", 5), "window_overlap": cfg.get("overlap", 7),
                         "window_shift_offset": cfg.get("shift_offset", 0), "seed": cfg.get("seed", 42),
                         "motion_translation_scale_factor": cfg.get("motion_translation_scale_factor", 0.0)
                        }
        except Exception as e:
            print(f"[Hunyuan Generator] Warning: Could not load embedded YAML config, using defaults. Error: {e}")
            defaults = { "height": 576, "width": 1024, "num_inference_steps": 25,
                         "min_appearance_guidance_scale": 1.0, "max_appearance_guidance_scale": 2.5,
                         "min_motion_guidance_scale": 1.0, "max_motion_guidance_scale": 3.5,
                         "fps_generation": 15, "motion_bucket_id": 127, "noise_aug_strength": 0.02,
                         "decode_chunk_size": 8, "i2i_noise_strength": 1.0, "n_sample_frames_batching": 14,
                         "pad_motion_frames": 5, "window_overlap": 7, "window_shift_offset": 0, "seed": 42,
                         "motion_translation_scale_factor": 0.0
                        }
        return {
            "required": {
                "hunyuan_models": ("HY_MODELS",),
                "preprocessed_data": ("PREPROCESSED_DATA",),
                "driving_data": ("DRIVING_DATA", {"forceInput": True}),
                "height": ("INT", {"default": defaults["height"], "min": 64, "max": 2048, "step": 64}),
                "width": ("INT", {"default": defaults["width"], "min": 64, "max": 2048, "step": 64}),
                "num_inference_steps": ("INT", {"default": defaults["num_inference_steps"], "min": 1, "max": 100}),
                "min_appearance_guidance_scale": ("FLOAT", {"default": defaults["min_appearance_guidance_scale"], "min": 1.0, "max": 20.0, "step": 0.1}),
                "max_appearance_guidance_scale": ("FLOAT", {"default": defaults["max_appearance_guidance_scale"], "min": 1.0, "max": 20.0, "step": 0.1}),
                "min_motion_guidance_scale": ("FLOAT", {"default": defaults["min_motion_guidance_scale"], "min": 1.0, "max": 20.0, "step": 0.1}),
                "max_motion_guidance_scale": ("FLOAT", {"default": defaults["max_motion_guidance_scale"], "min": 1.0, "max": 20.0, "step": 0.1}),
                "fps_generation": ("INT", {"default": defaults["fps_generation"], "min": 1, "max": 60}),
                "motion_bucket_id": ("INT", {"default": defaults["motion_bucket_id"], "min": 1, "max": 512}),
                "noise_aug_strength": ("FLOAT", {"default": defaults["noise_aug_strength"], "min": 0.0, "max": 1.0, "step": 0.001}),
                "decode_chunk_size": ("INT", {"default": defaults["decode_chunk_size"], "min": 1, "max": 100}),
                "i2i_noise_strength": ("FLOAT", {"default": defaults["i2i_noise_strength"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "n_sample_frames_batching": ("INT", {"default": defaults["n_sample_frames_batching"], "min": 1, "max": 64}),
                "pad_motion_frames": ("INT", {"default": defaults["pad_motion_frames"], "min": 0, "max": 32}),
                "window_overlap": ("INT", {"default": defaults["window_overlap"], "min": 0, "max": 64}),
                "window_shift_offset": ("INT", {"default": defaults["window_shift_offset"], "min": 0, "max": 64}),
                "seed": ("INT", {"default": defaults["seed"], "min": 0, "max": 0xffffffffffffffff, "widget":"seed"}),
                "motion_translation_scale_factor": ("FLOAT", {"default": defaults["motion_translation_scale_factor"], "min":0.0, "max":1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "HunyuanPortrait"

    def generate(self, hunyuan_models, preprocessed_data, driving_data,
                 height, width, num_inference_steps,
                 min_appearance_guidance_scale, max_appearance_guidance_scale,
                 min_motion_guidance_scale, max_motion_guidance_scale,
                 fps_generation, motion_bucket_id, noise_aug_strength,
                 decode_chunk_size, i2i_noise_strength,
                 n_sample_frames_batching, pad_motion_frames,
                 window_overlap, window_shift_offset, seed,
                 motion_translation_scale_factor
                 ):

        loaded_hy_models = hunyuan_models["models"]

        driving_video_path_to_clean = None
        temp_video_sub_dir_to_clean = None
        if driving_data and isinstance(driving_data, dict):
            driving_video_path_to_clean = driving_data.get("video_path")
            temp_video_sub_dir_to_clean = driving_data.get("temp_sub_dir")

        if preprocessed_data is None:
            if driving_video_path_to_clean and os.path.exists(driving_video_path_to_clean):
                try:
                    os.remove(driving_video_path_to_clean)
                    if temp_video_sub_dir_to_clean and os.path.exists(temp_video_sub_dir_to_clean) and not os.listdir(temp_video_sub_dir_to_clean):
                        os.rmdir(temp_video_sub_dir_to_clean)
                except OSError as e: print(f"Generator: Error cleaning temp video path '{driving_video_path_to_clean}' or its sub-directory: {e}")
            raise ValueError("Generator: Preprocessed data missing.")

        scheduler_instance = loaded_hy_models["scheduler"]
        device_for_calc = loaded_hy_models["device"]

        current_scheduler_config = scheduler_instance.config.copy()
        temp_scheduler_for_calc = type(scheduler_instance).from_config(current_scheduler_config)
        temp_scheduler_for_calc.set_timesteps(num_inference_steps, device=device_for_calc)

        init_timestep_val = min(int(num_inference_steps * i2i_noise_strength), num_inference_steps)
        t_start_val = max(num_inference_steps - init_timestep_val, 0)

        effective_timesteps_per_batch = temp_scheduler_for_calc.timesteps[t_start_val * temp_scheduler_for_calc.order :]
        num_diffusion_steps_per_batch = len(effective_timesteps_per_batch)
        if num_diffusion_steps_per_batch == 0 and num_inference_steps > 0:
            num_diffusion_steps_per_batch = 1

        num_preprocessed_frames = preprocessed_data['motion_pose_image'].shape[0]
        _num_frames_all_for_calc = num_preprocessed_frames + (pad_motion_frames * 2 if pad_motion_frames > 0 else 0)

        if (n_sample_frames_batching - window_overlap) <= 0:
            num_outer_loops = 1
        else:
            num_outer_loops = math.ceil(_num_frames_all_for_calc / (n_sample_frames_batching - window_overlap))

        total_diffusion_steps_for_pbar = num_outer_loops * num_diffusion_steps_per_batch
        if total_diffusion_steps_for_pbar == 0 and num_inference_steps > 0:
             total_diffusion_steps_for_pbar = num_diffusion_steps_per_batch if num_diffusion_steps_per_batch > 0 else 1

        pbar_comfy = comfy.utils.ProgressBar(total_diffusion_steps_for_pbar)

        processed_total_steps_for_pbar = 0
        def comfy_generator_callback(pipeline_instance, step_index, timestep, callback_kwargs):
            nonlocal processed_total_steps_for_pbar
            if processed_total_steps_for_pbar < total_diffusion_steps_for_pbar:
                 pbar_comfy.update(1)
                 processed_total_steps_for_pbar+=1
            return callback_kwargs

        try:
            seed_everything(seed)
            vae, unet, image_encoder, image_projector, pose_guider, motion_expression_model, \
            motion_headpose_model, motion_refiner, scheduler, device, dtype, pipeline_cfg_original = \
                (loaded_hy_models[k] for k in ["vae", "unet", "image_encoder", "image_projector",
                                             "pose_guider", "motion_expression", "motion_headpose",
                                             "motion_refiner", "scheduler", "device", "dtype", "pipeline_config"])

            sample = preprocessed_data
            ref_img = sample['ref_img'].unsqueeze(0).to(device, dtype=dtype)
            transformed_images = sample['transformed_images'].unsqueeze(0).to(device, dtype=dtype)
            arcface_embeddings = sample['arcface_embeddings']
            dwpose_images = sample['img_pose'].to(device, dtype=dtype)
            motion_pose_images = sample['motion_pose_image'].to(device, dtype=dtype)
            motion_face_images = sample['motion_face_image'].to(device, dtype=dtype)
            lmk_list = sample['lmk_list']

            pipe = HunyuanLongSVDPipeline(unet=unet, image_encoder=image_encoder, image_proj=image_projector,
                                          vae=vae, pose_guider=pose_guider, scheduler=scheduler)

            pose_cond_tensor_all, driven_feat_all, uncond_driven_feat_all, num_frames_all = [], [], [], 0
            from .vendored.dataset.utils import get_head_exp_motion_bucketid
            from einops import rearrange
            batch_motion = n_sample_frames_batching
            for idx in range(0, motion_pose_images.shape[0], batch_motion):
                mp_batch = motion_pose_images[idx:idx+batch_motion]
                mf_batch = motion_face_images[idx:idx+batch_motion]
                pc_batch = dwpose_images[idx:idx+batch_motion]
                lmks_b = lmk_list[idx:idx+batch_motion]
                nfb = mp_batch.shape[0]
                mbh_val, mbe_val = get_head_exp_motion_bucketid(lmks_b)
                mbh, mbe = torch.IntTensor([mbh_val]).to(device), torch.IntTensor([mbe_val]).to(device)

                mfeat = motion_expression_model(mf_batch)
                mfeat_embed = motion_refiner(mfeat, mbh, mbe)

                mp_batch_0_1_range = (mp_batch + 1.0) / 2.0
                mp_batch_for_headpose = mp_batch_0_1_range * 2.0 + 1.0
                dpfeat = motion_headpose_model(mp_batch_for_headpose)

                trans_scale = motion_translation_scale_factor
                dpfeat_embed = torch.cat([dpfeat['rotation'], dpfeat['translation'] * trans_scale], dim=-1)

                dfeat = torch.cat([mfeat_embed, dpfeat_embed.unsqueeze(1).repeat(1, mfeat_embed.shape[1], 1)], dim=-1).unsqueeze(0)
                udfeat = torch.zeros_like(dfeat)
                pc_batch_re = rearrange(pc_batch.unsqueeze(0), 'b f c h w -> b c f h w')
                pose_cond_tensor_all.append(pc_batch_re)
                driven_feat_all.append(dfeat)
                uncond_driven_feat_all.append(udfeat)
                num_frames_all += nfb

            pose_cond_tensor_all = torch.cat(pose_cond_tensor_all, dim=2)
            uncond_driven_feat_all = torch.cat(uncond_driven_feat_all, dim=1)
            driven_feat_all = torch.cat(driven_feat_all, dim=1)

            temp_pc_list, temp_df_list, temp_udf_list = [], [], []
            if pad_motion_frames > 0:
                for i_pad in range(pad_motion_frames):
                    w_pad = i_pad / pad_motion_frames
                    temp_pc_list.append(pose_cond_tensor_all[:,:,:1])
                    temp_df_list.append(driven_feat_all[:,:1]*w_pad)
                    temp_udf_list.append(uncond_driven_feat_all[:,:1])
            temp_pc_list.append(pose_cond_tensor_all)
            temp_df_list.append(driven_feat_all)
            temp_udf_list.append(uncond_driven_feat_all)
            if pad_motion_frames > 0:
                for i_pad in range(pad_motion_frames):
                    w_pad = i_pad / pad_motion_frames
                    temp_pc_list.append(pose_cond_tensor_all[:,:,-1:])
                    temp_df_list.append(driven_feat_all[:,-1:]*(1-w_pad))
                    temp_udf_list.append(uncond_driven_feat_all[:,:1])
            if pad_motion_frames > 0:
                pose_cond_tensor_all, driven_feat_all, uncond_driven_feat_all = \
                    torch.cat(temp_pc_list, dim=2), torch.cat(temp_df_list, dim=1), torch.cat(temp_udf_list, dim=1)

            effective_num_frames = num_frames_all + (pad_motion_frames * 2 if pad_motion_frames > 0 else 0)

            print("Generating...")

            video_frames_tensor = pipe(ref_image=ref_img.clone(), transformed_images=transformed_images.clone(),
                                     pose_cond_tensor=pose_cond_tensor_all, prompts=driven_feat_all, uncond_prompts=uncond_driven_feat_all,
                                     height=height, width=width, num_frames=effective_num_frames, decode_chunk_size=decode_chunk_size,
                                     motion_bucket_id=motion_bucket_id, fps=fps_generation, noise_aug_strength=noise_aug_strength,
                                     min_guidance_scale1=min_appearance_guidance_scale, max_guidance_scale1=max_appearance_guidance_scale,
                                     min_guidance_scale2=min_motion_guidance_scale, max_guidance_scale2=max_motion_guidance_scale,
                                     num_inference_steps=num_inference_steps, i2i_noise_strength=i2i_noise_strength,
                                     arcface_embeddings=arcface_embeddings, overlap=window_overlap, shift_offset=window_shift_offset,
                                     frames_per_batch=n_sample_frames_batching,
                                     device=device,
                                     callback_on_step_end=comfy_generator_callback,
                                     callback_on_step_end_tensor_inputs=["latents"]
                                     ).frames


            if pad_motion_frames > 0 and video_frames_tensor.shape[2] > (pad_motion_frames * 2):
                 video_frames_tensor = video_frames_tensor[:, :, pad_motion_frames:-pad_motion_frames, :, :]

            video_frames_output = video_frames_tensor.permute(0, 2, 3, 4, 1)
            if video_frames_output.shape[0] == 1:
                video_frames_output = video_frames_output.squeeze(0)
            else:
                print(f"Warning: HunyuanPortrait_Generator output batch size {video_frames_output.shape[0]} > 1. Outputting first video only for ComfyUI preview consistency.")
                video_frames_output = video_frames_output[0]

        finally:
            if driving_video_path_to_clean and os.path.exists(driving_video_path_to_clean):
                try:
                    os.remove(driving_video_path_to_clean)
                except OSError as e: print(f"Generator: Error cleaning temp video file '{driving_video_path_to_clean}': {e}")

            if temp_video_sub_dir_to_clean and os.path.exists(temp_video_sub_dir_to_clean) and not os.listdir(temp_video_sub_dir_to_clean):
                try:
                    os.rmdir(temp_video_sub_dir_to_clean)
                except OSError as e: print(f"Generator: Error cleaning temp video sub-directory '{temp_video_sub_dir_to_clean}': {e}")

        return (video_frames_output,)