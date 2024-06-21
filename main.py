import cv2
import os
import yaml
import numpy as np
from FisheyeCameraModel import FisheyeCameraModel
from BirdView import BirdView
import Utilities as utils

def get_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return None
    
    return frame

def process_image(images, data_path, output_path, camera_models, prefix):
    projected = []
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not open or find the image: {image_path}")
        img = camera_models[i].undistort(img)
        img = camera_models[i].project(img)
        img = camera_models[i].crop_and_flip(img)
        projected.append(img)
    
    birdview = BirdView()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)
    birdview.update_frames(projected)
    birdview.make_luminance_balance()
    birdview.stitch_all_parts()
    birdview.load_car_image(data_path)
    birdview.copy_car_image()
    # birdview.make_white_balance()

    print("Bird's eye view stitched image after added white balance")
    cv2.imshow("BirdView", birdview.getImage())
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_path, f'{prefix}-BEV.png'), birdview.getImage())

def process_imgvid(paths, data_path, output_path, camera_models, prefix):
    projected = []
    for i, path in enumerate(paths):
        img = get_first_frame(path)
        img = camera_models[i].undistort(img)
        img = camera_models[i].project(img)
        img = camera_models[i].crop_and_flip(img)
        projected.append(img)
    
    birdview = BirdView()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)
    birdview.update_frames(projected)
    birdview.make_luminance_balance()
    birdview.stitch_all_parts()
    birdview.load_car_image(data_path)
    birdview.copy_car_image()
    # birdview.make_white_balance()

    print("Bird's eye view stitched image after added white balance")
    # cv2.imshow("BirdView", birdview.getImage())
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(output_path, f'{prefix}-BEV.png'), birdview.getImage())

def process_video(input_videos, data_path, output_path, camera_models, prefix):
    frame_count = utils.find_minimum_frame_count(input_videos)-25
    caps = [cv2.VideoCapture(video) for video in input_videos]
    if not all(cap.isOpened() for cap in caps):
        print("Error: One or more video files couldn't be opened.")
        return
    first_left = True
    width = utils.init_constants()["config"]["total_w"]
    height = utils.init_constants()["config"]["total_h"]
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    # frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_path, f"{prefix}-BEV.mp4"), fourcc, fps, (width, height))
    outfront = cv2.VideoWriter(os.path.join(output_path, f"{prefix}-BEV-front.mp4"), fourcc, fps, utils.init_constants()["canvas_shape"]["front"])
    outback = cv2.VideoWriter(os.path.join(output_path, f"{prefix}-BEV-back.mp4"), fourcc, fps, utils.init_constants()["canvas_shape"]["back"])
    outleft = cv2.VideoWriter(os.path.join(output_path, f"{prefix}-BEV-left.mp4"), fourcc, fps, (utils.init_constants()["canvas_shape"]["left"][1], utils.init_constants()["canvas_shape"]["left"][0]))
    outright = cv2.VideoWriter(os.path.join(output_path, f"{prefix}-BEV-right.mp4"), fourcc, fps, (utils.init_constants()["canvas_shape"]["right"][1], utils.init_constants()["canvas_shape"]["right"][0]))

    birdview = BirdView()

    for i in range(frame_count):
        frames = []
        for cap in caps:
            if i == 1 and cap == caps[0]:
                first_left = False
                print("first left")
            if first_left and cap != caps[2]:
                for j in range(24):
                    cap.read()
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Couldn't read frame {i}.")
                break
            frames.append(frame)
        
        if len(frames) != len(camera_models):
            print("Error: Number of frames doesn't match number of camera models.")
            break

        projected = []
        for j, frame in enumerate(frames):
            img = camera_models[j].undistort(frame)
            img = camera_models[j].project(img)
            img = camera_models[j].crop_and_flip(img)
            projected.append(img)
        
        if i == 0:
            print("weights")
            Gmat, Mmat = birdview.get_weights_and_masks(projected)

        birdview.update_frames(projected)
        birdview.make_luminance_balance()
        birdview.stitch_all_parts()
        birdview.load_car_image(data_path)
        birdview.copy_car_image()

        birdview_height, birdview_width = birdview.getImage().shape[:2]

        if birdview_height != height or birdview_width != width:
            print(f"frame {i} is false")
        else:
            if i in [1, 10, 100, 200, frame_count - 1]:
                print("True")

        # cv2.imshow("BirdView", birdview.getImage())
        # cv2.waitKey(0)
        outfront.write(projected[0])
        outback.write(projected[1])
        outleft.write(projected[2])
        outright.write(projected[3])
        out.write(birdview.getImage())

    for cap in caps:
        cap.release()
    out.release()
    outfront.release()
    outback.release()
    outleft.release()
    outright.release()
    print(f"Output video saved to {os.path.join(output_path, f'{prefix}-BEV.mp4')}")

def process_stream(camera_model):
    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        print("Error: Could not open video stream from OBS Virtual Webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        desired_size = (960, 640)
        frame = cv2.resize(frame, desired_size)
        frame = camera_model.undistort(frame)
        frame = camera_model.project(frame)
        frame = camera_model.flip(frame)
        cv2.imshow("OBS Virtual Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = "video"
    for prefix in ["t1"]:
    # "t2", "t3", "t4", "t5"]:
        camera_names = ["front", "back", "left", "right"]
        yamls = []
        images = []
        videos = []
        data_path = "data"
        output_path = "data/output"
        debug=True
        if debug:
            print("Debug mode enabled")
            print(utils.init_constants()["config"])

        for name in camera_names:
            yamls.append(os.path.join(data_path, "yaml", f"{name}.yaml"))
            if mode == "video":
                videos.append(os.path.join(data_path, "videos", f"{prefix}-{name}.mp4"))
            if mode == "imgvid":
                videos.append(os.path.join(data_path, "videos", f"{prefix}-{name}.mp4"))
            else:
                images.append(os.path.join(data_path, "images", f"{name}-1.png"))

        camera_models = [FisheyeCameraModel(yaml_file, name, debug) for yaml_file, name in zip(yamls, camera_names)]

        selected_camera_model = camera_models[0]

        if mode == "video":
            process_video(videos, data_path, output_path, camera_models, prefix)
        if mode == "imgvid":
            process_imgvid(videos, data_path, output_path, camera_models, prefix)
        elif mode == "stream":
            process_stream(selected_camera_model)
        else:
            process_image(images, data_path, output_path, camera_models, prefix)
