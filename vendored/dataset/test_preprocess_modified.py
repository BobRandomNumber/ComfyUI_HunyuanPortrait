import cv2
import numpy as np
from PIL import Image
from skimage import transform as tf
import torch
import torchvision.transforms as transforms
import decord

from .utils import YoloFace

def get_dwpose(image_np_rgb):
    H, W = image_np_rgb.shape[:2]
    dwpose_image_canvas_np = np.zeros((H, W, 3), dtype=np.uint8)
    return dwpose_image_canvas_np

def align_face(image_np_rgb, landmark_np, output_shape=(112, 112)):
    points = np.asarray(landmark_np)
    dst_points = np.array([
        [30.2946 + 8.0, 51.6963], [65.5318 + 8.0, 51.5014], [48.0252 + 8.0, 71.7366],
        [33.5493 + 8.0, 92.3655], [62.7299 + 8.0, 92.2041]], dtype=np.float32)

    tform = tf.SimilarityTransform()
    if points.shape[0] != 5:
        print(f"Warning: Align_face received {points.shape[0]} landmarks, expected 5. Alignment might be poor or fail. Using fallback resize.")
        pil_img = Image.fromarray(image_np_rgb)
        resized_pil = pil_img.resize(output_shape, Image.Resampling.LANCZOS)
        return np.array(resized_pil).astype(np.float32) / 255.0

    tform.estimate(points, dst_points)

    image_float = image_np_rgb.astype(np.float32)
    if image_np_rgb.max() > 1.0:
        image_float = image_float / 255.0

    aligned_image_float = tf.warp(image_float,
                                  tform.inverse, output_shape=output_shape, mode='reflect', preserve_range=True)
    return aligned_image_float

def center_crop(img_driven_np_rgb, face_bbox_xyxy, scale=1.0):
    h_img, w_img = img_driven_np_rgb.shape[:2]
    x0, y0, x1, y1 = np.array(face_bbox_xyxy[:4]).astype(int)

    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    box_h, box_w = y1 - y0, x1 - x0

    if box_w <= 0 or box_h <=0:
        print(f"Warning: center_crop received invalid bbox [{x0},{y0},{x1},{y1}]. Returning original image.")
        return img_driven_np_rgb.copy()

    crop_side = int(max(box_w, box_h) * scale)

    new_x0 = center_x - crop_side // 2
    new_y0 = center_y - crop_side // 2
    new_x1 = new_x0 + crop_side
    new_y1 = new_y0 + crop_side

    pad_left = max(0, -new_x0)
    pad_top = max(0, -new_y0)
    pad_right = max(0, new_x1 - w_img)
    pad_bottom = max(0, new_y1 - h_img)

    crop_x0_valid = max(0, new_x0)
    crop_y0_valid = max(0, new_y0)
    crop_x1_valid = min(w_img, new_x1)
    crop_y1_valid = min(h_img, new_y1)

    if crop_x1_valid <= crop_x0_valid or crop_y1_valid <= crop_y0_valid:
        print(f"Warning: center_crop resulted in zero-size crop with bbox {face_bbox_xyxy} and scale {scale}. Using original bbox region or full image.")
        if img_driven_np_rgb[y0:y1, x0:x1].size > 0:
             img_cropped = img_driven_np_rgb[y0:y1, x0:x1].copy()
        else:
             img_cropped = img_driven_np_rgb.copy()
    else:
        img_cropped = img_driven_np_rgb[crop_y0_valid:crop_y1_valid, crop_x0_valid:crop_x1_valid]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img_cropped = cv2.copyMakeBorder(img_cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))

    return img_cropped

def process_bbox(bbox_xyxy, expand_radio, img_height, img_width):
    def expand(current_bbox_xyxy, ratio, h_img, w_img):
        b_x1, b_y1, b_x2, b_y2 = current_bbox_xyxy
        b_h = b_y2 - b_y1
        b_w = b_x2 - b_x1
        if b_w <=0 or b_h <=0: return current_bbox_xyxy

        expand_x1 = max(current_bbox_xyxy[0] - ratio * b_w, 0)
        expand_y1 = max(current_bbox_xyxy[1] - ratio * b_h, 0)
        expand_x2 = min(current_bbox_xyxy[2] + ratio * b_w, w_img)
        expand_y2 = min(current_bbox_xyxy[3] + ratio * b_h, h_img)
        return [expand_x1, expand_y1, expand_x2, expand_y2]

    def to_square(src_bbox_xyxy, exp_bbox_xyxy, h_img, w_img):
        exp_h = exp_bbox_xyxy[3] - exp_bbox_xyxy[1]
        exp_w = exp_bbox_xyxy[2] - exp_bbox_xyxy[0]
        if exp_w <=0 or exp_h <=0: return exp_bbox_xyxy

        c_h_exp = (exp_bbox_xyxy[1] + exp_bbox_xyxy[3]) / 2.0
        c_w_exp = (exp_bbox_xyxy[0] + exp_bbox_xyxy[2]) / 2.0

        c_h_src = (src_bbox_xyxy[1] + src_bbox_xyxy[3]) / 2.0
        c_w_src = (src_bbox_xyxy[0] + src_bbox_xyxy[2]) / 2.0

        shift_h, shift_w = 0.0, 0.0
        if exp_w < exp_h:
            delta = (exp_h - exp_w) / 2.0
            s_h_abs = min(delta, abs(c_h_src - c_h_exp))
            shift_h = s_h_abs if c_h_src > c_h_exp else -s_h_abs
        else:
            delta = (exp_w - exp_h) / 2.0
            s_w_abs = min(delta, abs(c_w_src - c_w_exp))
            shift_w = s_w_abs if c_w_src > c_w_exp else -s_w_abs

        c_h_sq = c_h_exp + shift_h
        c_w_sq = c_w_exp + shift_w

        half_side = min(exp_h, exp_w) / 2.0

        sq_x1 = max(0, c_w_sq - half_side)
        sq_y1 = max(0, c_h_sq - half_side)
        sq_x2 = min(w_img, c_w_sq + half_side)
        sq_y2 = min(h_img, c_h_sq + half_side)

        return [round(sq_x1), round(sq_y1), round(sq_x2), round(sq_y2)]

    bbox_expanded_xyxy = expand(bbox_xyxy, expand_radio, img_height, img_width)
    processed_bbox_xyxy = to_square(bbox_xyxy, bbox_expanded_xyxy, img_height, img_width)
    return processed_bbox_xyxy

def crop_resize_img(img_pil, bbox_xyxy, image_size):
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, min(x1, img_pil.width - 1))
    y1 = max(0, min(y1, img_pil.height - 1))
    x2 = max(x1 + 1, min(x2, img_pil.width))
    y2 = max(y1 + 1, min(y2, img_pil.height))

    if x1 >= x2 or y1 >= y2:
        print(f"Warning: Invalid crop bbox [{x1},{y1},{x2},{y2}] after clamping for image size {img_pil.size}. Returning resized original.")
        return img_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)

    img_cropped = img_pil.crop((x1, y1, x2, y2))
    img_resized = img_cropped.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img_resized

def crop_face_motion(image_np_rgb, landmark_np, motion_transform_fn, bbox_face_xyxy, scale_around_landmark=0.45):
    face_landmark = np.asarray(landmark_np)
    if face_landmark.ndim != 2 or face_landmark.shape[1] != 2 or face_landmark.shape[0] == 0:
        print("Warning: Invalid or empty landmarks for crop_face_motion. Using full bbox_face_xyxy for motion crop.")
        final_crop_x_min, final_crop_y_min, final_crop_x_max, final_crop_y_max = bbox_face_xyxy
    else:
        face_x_min_lmk, face_x_max_lmk = min(face_landmark[:, 0]), max(face_landmark[:, 0])
        face_y_min_lmk, face_y_max_lmk = min(face_landmark[:, 1]), max(face_landmark[:, 1])

        box_x_min, box_y_min, box_x_max, box_y_max = bbox_face_xyxy

        final_crop_x_min = max(box_x_min, face_x_min_lmk - (face_x_min_lmk - box_x_min) * scale_around_landmark)
        final_crop_x_max = min(box_x_max, face_x_max_lmk + (box_x_max - face_x_max_lmk) * scale_around_landmark)
        final_crop_y_min = max(box_y_min, face_y_min_lmk - (face_y_min_lmk - box_y_min) * scale_around_landmark)
        final_crop_y_max = min(box_y_max, face_y_max_lmk + (box_y_max - face_y_max_lmk) * scale_around_landmark)

    lmk_expanded_bbox_xyxy = [final_crop_x_min, final_crop_y_min, final_crop_x_max, final_crop_y_max]

    motion_crop_face_np = center_crop(image_np_rgb.copy(), lmk_expanded_bbox_xyxy, scale=1.0)
    motion_crop_face_pil = Image.fromarray(motion_crop_face_np)
    motion_crop_face_tensor = motion_transform_fn(motion_crop_face_pil)
    return motion_crop_face_tensor

def preprocess_modified(
    source_image_pil: Image.Image,
    video_path: str,
    yolo_detector_instance: YoloFace,
    arcface_session_instance,
    limit: int,
    output_image_size: int,
    area_scale: float,
    use_arcface: bool,
    progress_callback=None
):
    img_size_dino = (224, 224)
    img_transform_dino = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(img_size_dino, scale=(1.0, 1.0), ratio=(1.0, 1.0), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False),
    ])
    to_tensor_normalize_half = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pose_to_tensor = transforms.Compose([transforms.ToTensor()])
    motion_transform = transforms.Compose([
        transforms.Resize(256), transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Results in [-1, 1] for [0,1] input PIL
    ])

    align_instance_detect_fn = yolo_detector_instance.detect
    imSrc_pil_rgb = source_image_pil.convert('RGB')
    origin_src_np_rgb = np.array(imSrc_pil_rgb)

    landmarks_orig_list, _, bboxes_orig_xywh_list = align_instance_detect_fn(origin_src_np_rgb[:,:,::-1].copy())
    if not bboxes_orig_xywh_list.size:
        raise RuntimeError("No face detected in the source image by YoloFace.")

    def get_box_area_xywh(box_xywh): return box_xywh[2] * box_xywh[3]
    areas_orig = np.array([get_box_area_xywh(bbox) for bbox in bboxes_orig_xywh_list])
    max_idx_orig = np.argmax(areas_orig)
    bbox_orig_xywh = bboxes_orig_xywh_list[max_idx_orig]
    x1_o, y1_o, w_o, h_o = bbox_orig_xywh
    bbox_orig_xyxy = [x1_o, y1_o, x1_o + w_o, y1_o + h_o]

    crop_bbox_processed_xyxy = process_bbox(bbox_orig_xyxy, 1.0, origin_src_np_rgb.shape[0], origin_src_np_rgb.shape[1])
    imSrc_cropped_resized_pil = crop_resize_img(imSrc_pil_rgb, crop_bbox_processed_xyxy, output_image_size)
    imSrc_cropped_resized_np_rgb = np.array(imSrc_cropped_resized_pil)

    arcface_onnx_input_size = (112, 112)
    landmarks_on_cropped_list, _, bboxes_on_cropped_xywh_list = align_instance_detect_fn(imSrc_cropped_resized_np_rgb[:,:,::-1].copy())

    arcface_image_for_embedding_np_uint8 = None
    if use_arcface and landmarks_on_cropped_list.size > 0:
        areas_on_cropped = np.array([get_box_area_xywh(bbox) for bbox in bboxes_on_cropped_xywh_list])
        max_idx_on_cropped = np.argmax(areas_on_cropped)
        landmarks_for_align = landmarks_on_cropped_list[max_idx_on_cropped]

        arcface_image_aligned_float_np = align_face(imSrc_cropped_resized_np_rgb, landmarks_for_align, output_shape=arcface_onnx_input_size)
        arcface_image_for_embedding_np_uint8 = (arcface_image_aligned_float_np * 255).astype(np.uint8)
    elif use_arcface:
        print("Warning: No face/landmarks on cropped source for ArcFace. Resizing unaligned crop to 112x112 for embedding.")
        temp_pil = Image.fromarray(imSrc_cropped_resized_np_rgb)
        arcface_image_for_embedding_np_uint8 = np.array(temp_pil.resize(arcface_onnx_input_size, Image.Resampling.LANCZOS))

    ref_img_tensor = to_tensor_normalize_half(imSrc_cropped_resized_pil)
    transformed_images_dino_tensor = img_transform_dino(imSrc_cropped_resized_pil)

    arcface_embeddings_source = np.zeros((1, 512), dtype=np.float32)
    if use_arcface and arcface_session_instance is not None and arcface_image_for_embedding_np_uint8 is not None:
        arcface_input_onnx = arcface_image_for_embedding_np_uint8.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
        arcface_embeddings_source = arcface_session_instance.run(None, {"data": arcface_input_onnx})[0]
        arcface_embeddings_source = arcface_embeddings_source / np.linalg.norm(arcface_embeddings_source, axis=1, keepdims=True)

    cap = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames_video = len(cap)
    frames_to_process_actual = min(total_frames_video, limit)

    driven_dwpose_images_list, motion_face_images_list, motion_pose_images_list = [], [], []
    driven_image_vae_list, lmk_list_driving = [], []

    first_frame_np_rgb = cap[0].asnumpy()
    landmarks_drv_first_list, _, bboxes_drv_first_xywh_list = align_instance_detect_fn(first_frame_np_rgb[:,:,::-1].copy())

    if not bboxes_drv_first_xywh_list.size:
        print("Warning: No face in first frame of driving video. Using scaled source bbox as guide.")
        h_drv, w_drv = first_frame_np_rgb.shape[:2]; h_src, w_src = origin_src_np_rgb.shape[:2]
        scale_h, scale_w = h_drv/h_src if h_src > 0 else 1.0, w_drv/w_src if w_src > 0 else 1.0
        x1_s, y1_s, w_s, h_s = bbox_orig_xywh
        bbox_drv_first_xyxy = [x1_s*scale_w, y1_s*scale_h, (x1_s+w_s)*scale_w, (y1_s+h_s)*scale_h]
    else:
        areas_drv_first = np.array([get_box_area_xywh(bbox) for bbox in bboxes_drv_first_xywh_list])
        max_idx_drv_first = np.argmax(areas_drv_first)
        bbox_drv_first_xywh = bboxes_drv_first_xywh_list[max_idx_drv_first]
        x1df, y1df, wdf, hdf = bbox_drv_first_xywh
        bbox_drv_first_xyxy = [x1df, y1df, x1df+wdf, y1df+hdf]

    bbox_s_driving_xyxy = process_bbox(bbox_drv_first_xyxy, 1.0, first_frame_np_rgb.shape[0], first_frame_np_rgb.shape[1])

    fallback_landmark_drv = None
    if landmarks_drv_first_list.size > 0 and 'max_idx_drv_first' in locals():
        fallback_landmark_drv = landmarks_drv_first_list[max_idx_drv_first]
    elif landmarks_on_cropped_list.size > 0 and 'max_idx_on_cropped' in locals():
        max_idx_on_cropped = np.argmax(np.array([get_box_area_xywh(bbox) for bbox in bboxes_on_cropped_xywh_list]))
        print("Warning: No landmarks in first driving video frame, trying source landmarks as initial fallback.")
        fallback_landmark_drv = landmarks_on_cropped_list[max_idx_on_cropped]


    for drive_idx in range(frames_to_process_actual):
        frame_np_rgb = cap[drive_idx].asnumpy()
        current_frame_landmarks_list, _, current_frame_bboxes_xywh_list = align_instance_detect_fn(frame_np_rgb[:,:,::-1].copy())

        current_frame_landmarks_for_crop = np.array([])
        current_frame_main_bbox_xyxy_for_crop = bbox_s_driving_xyxy

        if current_frame_landmarks_list.size > 0:
            areas_curr = np.array([get_box_area_xywh(bbox) for bbox in current_frame_bboxes_xywh_list])
            max_idx_curr = np.argmax(areas_curr)
            current_frame_landmarks_for_crop = current_frame_landmarks_list[max_idx_curr]
            xcf, ycf, wcf, hcf = current_frame_bboxes_xywh_list[max_idx_curr]
            current_frame_main_bbox_xyxy_for_crop = [xcf, ycf, xcf+wcf, ycf+hcf]
            fallback_landmark_drv = current_frame_landmarks_for_crop
        elif fallback_landmark_drv is not None:
            print(f"Warning: No landmarks in driving frame {drive_idx}. Using fallback landmarks for motion crop.")
            current_frame_landmarks_for_crop = fallback_landmark_drv

        valid_lmks_for_bucket = current_frame_landmarks_for_crop if current_frame_landmarks_for_crop.size > 0 and current_frame_landmarks_for_crop.shape == (5,2) else np.zeros((5,2))
        lmk_list_driving.append(valid_lmks_for_bucket)

        motion_face_tensor = crop_face_motion(frame_np_rgb.copy(), current_frame_landmarks_for_crop, motion_transform, current_frame_main_bbox_xyxy_for_crop)
        motion_face_images_list.append(motion_face_tensor)

        driving_frame_pil = Image.fromarray(frame_np_rgb)
        driven_pose_pil_cropped = crop_resize_img(driving_frame_pil, bbox_s_driving_xyxy, output_image_size)

        motion_pose_tensor = motion_transform(driven_pose_pil_cropped)
        motion_pose_images_list.append(motion_pose_tensor)

        dwpose_output_np = get_dwpose(np.array(driven_pose_pil_cropped)) # Uses black image
        driven_dwpose_images_list.append(pose_to_tensor(dwpose_output_np))

        driven_image_vae_list.append(to_tensor_normalize_half(driven_pose_pil_cropped))

        if progress_callback:
            progress_callback()

    if not motion_face_images_list:
        raise ValueError("No frames processed from driving video. Check video path, content, and landmark detection.")

    sample_dict = dict(
        ref_img=ref_img_tensor, transformed_images=transformed_images_dino_tensor,
        arcface_embeddings=arcface_embeddings_source,
        img_pose=torch.stack(driven_dwpose_images_list, dim=0),
        motion_face_image=torch.stack(motion_face_images_list, dim=0),
        motion_pose_image=torch.stack(motion_pose_images_list, dim=0),
        driven_image=torch.stack(driven_image_vae_list, dim=0),
        lmk_list=lmk_list_driving,
    )
    return sample_dict