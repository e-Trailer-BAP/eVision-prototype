import cv2
import numpy as np
import yaml

def init_constants():
    yaml_file = 'data/yaml/config.yaml'
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    scale = config['scale']
    shift_w = config['shift_w']
    shift_h = config['shift_h']
    total_w = config['car_width'] + 2 * shift_w
    total_h = config['car_length'] + 2 * shift_h
    xl = shift_w
    xr = total_w - xl
    yt = shift_h
    yb = total_h - yt

    project_shapes = {
        "canvas_shape":{
            "front": (total_w, yt),
            "back": (total_w, yt),
            "left": (total_h, xl),
            "right": (total_h, xl)
            },
        "dXdY":{
            "front": (total_w/2, shift_h),
            "back": (total_w/2, shift_h),
            "left": (total_h/2, shift_w),
            "right": (total_h/2, shift_w)
        },
        "config":{
            "shift_w": shift_w,
            "shift_h": shift_h,
            "scale": scale,
            "total_w": total_w,
            "total_h": total_h,

        }
    }

    return project_shapes

def convert_binary_to_bool(mask):
    bool_mask = mask.astype(np.float32) / 255.0
    return bool_mask

def adjust_luminance(gray, factor):
    adjusted = gray.astype(np.float32) * factor
    return adjusted

def get_mean_statistic(gray, mask):
    return np.sum(gray * mask)

def mean_luminance_ratio(grayA, grayB, mask):
    return get_mean_statistic(grayA, mask) / get_mean_statistic(grayB, mask)

def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask

def get_overlap_region_mask(imA, imB):
    overlap = cv2.bitwise_and(imA, imB)
    mask = get_mask(overlap)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return dilated_mask

def get_outmost_polygon_boundary(img):
    mask = get_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest_contour, 0.009 * cv2.arcLength(largest_contour, True), True)
    return polygon

def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    overlap_mask = get_overlap_region_mask(imA, imB)
    overlap_mask_inv = cv2.bitwise_not(overlap_mask)
    imA_diff = cv2.bitwise_and(imA, imA, mask=overlap_mask_inv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlap_mask_inv)
    G = get_mask(imA).astype(np.float32) / 255.0
    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)

    for y in range(overlap_mask.shape[0]):
        for x in range(overlap_mask.shape[1]):
            if overlap_mask[y, x] == 255:
                pt = (x, y)
                dist_to_B = cv2.pointPolygonTest(polyB, pt, True)
                if dist_to_B < dist_threshold:
                    dist_to_A = cv2.pointPolygonTest(polyA, pt, True)
                    dist_to_B = dist_to_B ** 2
                    dist_to_A = dist_to_A ** 2
                    G[y, x] = dist_to_B / (dist_to_A + dist_to_B)

    return G, overlap_mask

def make_white_balance(image):
    channels = cv2.split(image)
    means = cv2.mean(image)
    K = sum(means[:3]) / 3.0
    channels[0] = adjust_luminance(channels[0], K / means[0])
    channels[1] = adjust_luminance(channels[1], K / means[1])
    channels[2] = adjust_luminance(channels[2], K / means[2])
    balanced_image = cv2.merge(channels)
    return balanced_image

def get_frame_count(video_path):
    """Return the frame count of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return float('inf')
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def find_minimum_frame_count(video_paths):
    """Return the video path with the minimum frame count and the frame count."""
    min_frame_count = float('inf')
    
    for path in video_paths:
        frame_count = get_frame_count(path)
        if frame_count < min_frame_count:
            min_frame_count = frame_count
    
    return  min_frame_count