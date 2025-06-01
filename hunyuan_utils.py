import os
import json
import torch
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf
from PIL import Image

from comfy.model_management import get_torch_device
from comfy.utils import load_torch_file

from .vendored.models.condition.unet_3d_svd_condition_ip import UNet3DConditionSVDModel, init_ip_adapters
from .vendored.models.condition.pose_guider import PoseGuider
from .vendored.models.condition.coarse_motion import HeadExpression, HeadPose
from .vendored.models.condition.refine_motion import IntensityAwareMotionRefiner
from .vendored.models.dinov2.models.vision_transformer import vit_large, ImageProjector
from .vendored.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler as HunyuanEulerDiscreteScheduler
from .vendored.dataset.utils import YoloFace

from diffusers import AutoencoderKLTemporalDecoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "embedded_configs")

def load_embedded_yaml_config():
    yaml_path = os.path.join(CONFIG_DIR, "hunyuan-portrait.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Hunyuan Portrait YAML config not found at {yaml_path}")
    return OmegaConf.load(yaml_path)

def load_embedded_json_config(filename):
    json_path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON config not found at {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

def load_hunyuan_models(
    vae_path, unet_path, image_encoder_path, image_projector_path,
    pose_guider_path, motion_expression_path, motion_headpose_path, motion_refiner_path,
    dtype_str="fp16", device_str="cuda"
):
    target_device = torch.device(device_str if torch.cuda.is_available() and device_str=="cuda" else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Hunyuan Models: CUDA selected but not available, falling back to CPU.")

    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    if dtype_str == "bf16":
        try:
            dtype = torch.bfloat16
        except AttributeError:
            print("Hunyuan Models: bfloat16 not supported on this PyTorch version. Falling back to float32.")
            dtype = torch.float32

    cpu_device_obj = torch.device("cpu")
    pipeline_cfg = load_embedded_yaml_config()

    vae_config = load_embedded_json_config("vae_config.json")
    vae = AutoencoderKLTemporalDecoder.from_config(vae_config)
    vae_weights = load_torch_file(vae_path, device=cpu_device_obj)
    vae.load_state_dict(vae_weights)
    vae.to(dtype=dtype, device=target_device).eval()

    unet_config_data = load_embedded_json_config("unet_config.json")
    unet_init_kwargs = {
        "sample_size": None,
        "in_channels": pipeline_cfg.get("unet_in_channels", unet_config_data.get("in_channels", 8)),
        "out_channels": unet_config_data.get("out_channels", 4),
        "down_block_types": tuple(unet_config_data.get("down_block_types")),
        "up_block_types": tuple(unet_config_data.get("up_block_types")),
        "block_out_channels": tuple(unet_config_data.get("block_out_channels")),
        "layers_per_block": unet_config_data.get("layers_per_block", 2),
        "cross_attention_dim": unet_config_data.get("cross_attention_dim", 1024),
        "addition_time_embed_dim": pipeline_cfg.get("addition_time_embed_dim", 256),
        "projection_class_embeddings_input_dim": pipeline_cfg.get("projection_class_embeddings_input_dim"),
        "num_attention_heads": tuple(unet_config_data.get("num_attention_heads", (5,10,20,20))),
        "transformer_layers_per_block": unet_config_data.get("transformer_layers_per_block", 1),
        "num_frames": pipeline_cfg.get("num_frames_for_unet_config", 25)
    }
    if unet_init_kwargs["projection_class_embeddings_input_dim"] is None:
        unet_init_kwargs["projection_class_embeddings_input_dim"] = 768
    unet = UNet3DConditionSVDModel(**unet_init_kwargs)
    init_ip_adapters(unet, pipeline_cfg.get("num_adapter_embeds", [4,4]), pipeline_cfg.get("ip_motion_scale",1.0))
    unet_weights = load_torch_file(unet_path, device=cpu_device_obj)
    unet.load_state_dict(unet_weights)
    unet.to(dtype=dtype, device=target_device).eval()

    image_encoder = vit_large(
        patch_size=pipeline_cfg.get("dino_patch_size", 14),
        num_register_tokens=pipeline_cfg.get("dino_num_register_tokens", 4),
        img_size=pipeline_cfg.get("dino_img_size", 526),
        init_values=pipeline_cfg.get("dino_init_values", 1.0),
        block_chunks=pipeline_cfg.get("dino_block_chunks", 0),
        backbone=True, layers_output=True,
        add_adapter_layer=list(pipeline_cfg.get("dino_add_adapter_layer", [3, 7, 11, 15, 19, 23])),
        visual_adapter_dim=pipeline_cfg.get("dino_visual_adapter_dim", 384),
    )
    image_encoder_weights = load_torch_file(image_encoder_path, device=cpu_device_obj)
    image_encoder.load_state_dict(image_encoder_weights, strict=True)
    image_encoder.to(dtype=dtype, device=target_device).eval()

    dino_input_size_for_proj_calc = 224
    patch_size_for_proj_calc = pipeline_cfg.get("dino_patch_size", 14)
    num_registers_for_proj_calc = pipeline_cfg.get("dino_num_register_tokens", 4)
    num_cls_token_for_proj_calc = 1
    num_patch_tokens_for_proj = (dino_input_size_for_proj_calc // patch_size_for_proj_calc) ** 2
    projector_input_features = num_patch_tokens_for_proj + num_cls_token_for_proj_calc + num_registers_for_proj_calc
    projector_output_features = pipeline_cfg.get("num_queries", 64)

    image_projector = ImageProjector(in_features=projector_input_features, out_features=projector_output_features)
    image_projector_weights = load_torch_file(image_projector_path, device=cpu_device_obj)
    if "linear.weight" not in image_projector_weights and ("weight" in image_projector_weights or "bias" in image_projector_weights):
        image_projector_weights = {"linear.weight": image_projector_weights.get("weight"), "linear.bias": image_projector_weights.get("bias")}
    image_projector.load_state_dict(image_projector_weights, strict=True)
    image_projector.to(dtype=dtype, device=target_device).eval()

    pose_guider = PoseGuider(
        conditioning_embedding_channels=pipeline_cfg.get("pose_guider_cond_embed_channels", 320),
        block_out_channels=tuple(pipeline_cfg.get("pose_guider_block_out_channels", (16, 32, 96, 256)))
    )
    pose_guider_weights = load_torch_file(pose_guider_path, device=cpu_device_obj)
    pose_guider.load_state_dict(pose_guider_weights)
    pose_guider.to(dtype=dtype, device=target_device).eval()

    motion_expression_model = HeadExpression(out_feat_dim=pipeline_cfg.get("input_expression_dim", 512))
    motion_expression_weights = load_torch_file(motion_expression_path, device=cpu_device_obj)
    motion_expression_model.load_state_dict(motion_expression_weights)
    motion_expression_model.to(dtype=dtype, device=target_device).eval()

    motion_headpose_model = HeadPose()
    motion_headpose_weights = load_torch_file(motion_headpose_path, device=cpu_device_obj)
    motion_headpose_model.load_state_dict(motion_headpose_weights)
    motion_headpose_model.to(dtype=dtype, device=target_device).eval()

    motion_refiner = IntensityAwareMotionRefiner(
        input_dim=pipeline_cfg.get("input_expression_dim", 512),
        output_dim=pipeline_cfg.get("motion_expression_dim", 1024),
        num_queries=pipeline_cfg.get("num_queries_motion_refiner", 64),
        intensity_embed_dim=pipeline_cfg.get("motion_intensity_embed_dim",768),
        width=pipeline_cfg.get("motion_refiner_width",768),
        layers=pipeline_cfg.get("motion_refiner_layers",6),
        heads=pipeline_cfg.get("motion_refiner_heads",8)
    )
    motion_refiner_weights = load_torch_file(motion_refiner_path, device=cpu_device_obj)
    motion_refiner.load_state_dict(motion_refiner_weights)
    motion_refiner.to(dtype=dtype, device=target_device).eval()

    scheduler_config_raw = load_embedded_json_config("scheduler_config.json")
    final_scheduler_args = {}
    final_scheduler_args["num_train_timesteps"] = int(scheduler_config_raw.get("num_train_timesteps", 1000))
    final_scheduler_args["beta_start"] = float(scheduler_config_raw.get("beta_start", 0.00085))
    final_scheduler_args["beta_end"] = float(scheduler_config_raw.get("beta_end", 0.012))
    final_scheduler_args["beta_schedule"] = str(scheduler_config_raw.get("beta_schedule", "scaled_linear"))
    trained_betas_val = scheduler_config_raw.get("trained_betas", None)
    if isinstance(trained_betas_val, list): final_scheduler_args["trained_betas"] = np.array(trained_betas_val, dtype=np.float32)
    else: final_scheduler_args["trained_betas"] = trained_betas_val
    final_scheduler_args["prediction_type"] = str(scheduler_config_raw.get("prediction_type", "v_prediction"))
    final_scheduler_args["interpolation_type"] = str(scheduler_config_raw.get("interpolation_type", "linear"))

    val_k = scheduler_config_raw.get("use_karras_sigmas", True)
    if isinstance(val_k, str): final_scheduler_args["use_karras_sigmas"] = val_k.lower() == "true"
    elif isinstance(val_k, bool): final_scheduler_args["use_karras_sigmas"] = val_k
    else: final_scheduler_args["use_karras_sigmas"] = False

    sigma_min_val = scheduler_config_raw.get("sigma_min")
    if sigma_min_val is not None: final_scheduler_args["sigma_min"] = float(sigma_min_val)
    sigma_max_val = scheduler_config_raw.get("sigma_max")
    if sigma_max_val is not None: final_scheduler_args["sigma_max"] = float(sigma_max_val)

    final_scheduler_args["timestep_spacing"] = str(scheduler_config_raw.get("timestep_spacing", "leading"))
    final_scheduler_args["timestep_type"] = str(scheduler_config_raw.get("timestep_type", "continuous"))

    steps_offset_val = scheduler_config_raw.get("steps_offset")
    if steps_offset_val is not None: final_scheduler_args["steps_offset"] = int(steps_offset_val)

    def coerce_bool(key, default_val):
        val = scheduler_config_raw.get(key, default_val)
        if isinstance(val, str): return val.lower() == "true"
        if isinstance(val, bool): return val
        return default_val

    final_scheduler_args["rescale_betas_zero_snr"] = coerce_bool("rescale_betas_zero_snr", False)
    final_scheduler_args["set_alpha_to_one"] = coerce_bool("set_alpha_to_one", False)
    final_scheduler_args["clip_sample"] = coerce_bool("clip_sample", False)
    final_scheduler_args["skip_prk_steps"] = coerce_bool("skip_prk_steps", True)

    if "use_beta_sigmas" in final_scheduler_args:
        del final_scheduler_args["use_beta_sigmas"]
    if "use_exponential_sigmas" in final_scheduler_args:
        del final_scheduler_args["use_exponential_sigmas"]

    if "_class_name" in scheduler_config_raw: final_scheduler_args["_class_name"] = scheduler_config_raw["_class_name"]
    if "_diffusers_version" in scheduler_config_raw: final_scheduler_args["_diffusers_version"] = scheduler_config_raw["_diffusers_version"]

    scheduler = HunyuanEulerDiscreteScheduler.from_config(config=final_scheduler_args)

    print("HunyuanPortrait Models Loaded.")

    models_bundle = {
        "vae": vae, "unet": unet, "image_encoder": image_encoder, "image_projector": image_projector,
        "pose_guider": pose_guider, "motion_expression": motion_expression_model,
        "motion_headpose": motion_headpose_model, "motion_refiner": motion_refiner,
        "scheduler": scheduler, "pipeline_config": pipeline_cfg,
        "device": target_device, "dtype": dtype
    }
    return models_bundle

def load_onnx_utils(arcface_model_path, yoloface_model_path, device_str="cuda"):
    yolo_device_str_for_yoloface_class = device_str if torch.cuda.is_available() and device_str=="cuda" else "cpu"

    arcface_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device_str == "cuda" and torch.cuda.is_available() else ['CPUExecutionProvider']

    try:
        arcface_session = ort.InferenceSession(arcface_model_path, providers=arcface_providers)
    except Exception as e:
        print(f"ONNX Utils: Failed to load ArcFace model with providers {arcface_providers}. Error: {e}")
        print("ONNX Utils: Attempting with CPUExecutionProvider only.")
        arcface_session = ort.InferenceSession(arcface_model_path, providers=['CPUExecutionProvider'])

    yoloface_detector = YoloFace(pt_path=yoloface_model_path, device=yolo_device_str_for_yoloface_class)

    return {"arcface_session": arcface_session, "yoloface_detector": yoloface_detector}

def hunyuan_comfy_preprocess(
    source_image_pil: Image.Image,
    video_path_for_preprocessing: str,
    onnx_models,
    pipeline_cfg, limit_frames: int, output_image_size: int,
    dwpose_area_scale: float, use_arcface_bool: bool,
    progress_callback=None
):
    yolo_detector_instance = onnx_models["yoloface_detector"]
    arcface_session_instance = onnx_models["arcface_session"]

    try:
        from .vendored.dataset.test_preprocess_modified import preprocess_modified
    except ImportError as e:
        print(f"ImportError for modified preprocessor: {e}")
        raise NotImplementedError("Modified preprocessing script 'test_preprocess_modified.py' is essential and not found in 'vendored/dataset/'. Ensure it exists and is correctly placed.")

    sample_dict = preprocess_modified(
        source_image_pil=source_image_pil,
        video_path=video_path_for_preprocessing,
        yolo_detector_instance=yolo_detector_instance,
        arcface_session_instance=arcface_session_instance,
        limit=limit_frames,
        output_image_size=output_image_size,
        area_scale=dwpose_area_scale,
        use_arcface=use_arcface_bool,
        progress_callback=progress_callback
    )
    return sample_dict