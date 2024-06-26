
# eVision-prototype
Prototype of the surround view system

# Codebase

This codebase is designed to create a bird's eye view (BEV) image or video from fisheye camera inputs mounted on a vehicle. It processes video streams or images from multiple cameras (front, back, left, and right), undistorts and projects the images, and then stitches them together to create a comprehensive top-down view.

## Files

### `main.py`
This is the main entry point for the application. It contains functions to process images, videos, or streams to generate bird's eye view images.

Key Functions:
- `get_first_frame(video_path)`: Captures the first frame from a video file.
- `process_image(images, data_path, output_path, camera_models, prefix)`: Processes a list of images.
- `process_imgvid(paths, data_path, output_path, camera_models, prefix)`: Processes first frame of video files.
- `process_video(input_videos, data_path, output_path, camera_models, prefix)`: Processes video files and stitches frames together.
- `process_stream(camera_model)`: Processes a live video stream from a camera.

### `FisheyeCameraModel.py`
This module handles the fisheye camera model. It reads camera parameters, undistorts images, and projects them for stitching.

Key Methods:
- `__init__(self, camera_param_file, camera_name, debug=False, scale_xy=(1.0, 1.0), shift_xy=(0, 0))`: Initializes the camera model.
- `load_camera_params()`: Loads camera parameters from a YAML file.
- `update_undistort_maps()`: Updates the undistortion maps based on camera parameters.
- `undistort(image)`: Undistorts an input image.
- `project(image)`: Projects an undistorted image.
- `crop_and_flip(image)`: Crops and flips the image based on the camera position.
- `save_data()`: Saves updated camera parameters back to a file.
**To-Do**: make save data 

### `BirdView.py`
This module handles the creation of the bird's eye view image by merging images from different cameras.

Key Methods:
- `__init__(self)`: Initializes the bird's eye view object.
- `update_frames(self, new_frames)`: Updates the frames from different cameras.
- `merge(self, imA, imB, k)`: Merges two images using a weight mask.
- `stitch_all_parts(self)`: Stitches all parts of the bird's eye view together.
- `load_car_image(self, data_path)`: Loads an image of the car to overlay on the bird's eye view.
- `copy_car_image(self)`: Copies the car image onto the stitched image.
- `make_luminance_balance(self)`: Balances the luminance of the stitched image.
- `get_weights_and_masks(self, images)`: Gets the weight masks for merging images.
- `make_white_balance(self)`: Applies white balance to the final image.
- `getImage(self)`: Returns the final stitched bird's eye view image.

### `Utilities.py`
This module contains utility functions used throughout the project.

Key Functions:
- `init_constants()`: Initializes constants used for projection and stitching.
- `convert_binary_to_bool(mask)`: Converts a binary mask to a boolean mask.
- `adjust_luminance(gray, factor)`: Adjusts the luminance of a grayscale image.
- `get_mean_statistic(gray, mask)`: Calculates the mean statistic of a masked region.
- `mean_luminance_ratio(grayA, grayB, mask)`: Calculates the luminance ratio between two grayscale images.
- `get_mask(img)`: Generates a binary mask from an image.
- `get_overlap_region_mask(imA, imB)`: Generates a mask for the overlapping region between two images.
- `get_outmost_polygon_boundary(img)`: Finds the outmost polygon boundary of a mask.
- `get_weight_mask_matrix(imA, imB, dist_threshold=5)`: Creates a weight mask matrix for blending two images.
- `make_white_balance(image)`: Adjusts the white balance of an image.
- `get_frame_count(video_path)`: Returns the frame count of a video file.
- `find_minimum_frame_count(video_paths)`: Finds the video with the minimum frame count from a list of videos.

### `undistort.py` 
This module is responsible for undistorting fisheye camera images, detecting centroids for perspective transformation, and projecting the images onto a common plane. 
Key Functions: 
- `process_image(images, data_path, output_path, camera_models, prefix)`: Processes a list of images by undistorting and projecting them. 
- `find_centroids(camera_name, Undistorted, HRL=30, SRL=160, VRL=20, HRU=12, SRU=40, VRU=40)`: Finds centroids of detected features in an undistorted image. 
- `order_centroids(centroids)`: Orders the detected centroids for perspective transformation. - `project(image, camera_name, centroid)`: Projects an undistorted image using the calculated centroids and perspective transformation matrix. 
**Usage**: This script can be run independently to preprocess calibration images for the fisheye cameras. It detects the centroids, orders them, and applies the perspective transformation to generate the projected images and matrices. 
**To Do**: Write the generated projection matrix to a YAML

### `projection.py` 
This module handles the perspective transformation of an image based on predefined corner points and their corresponding real-world coordinates. 
Key Steps: 
1. Load an undistorted image. 
2. Define the coordinates of four corners in the image. 
3. Define the corresponding real-world coordinates. 
4. Compute the perspective transformation matrix. 
5.  Apply the perspective transformation to the image. 
**Usage**: This script can be used to walkthrough the process of creating a projection matrix step-by-step.

### `threadrecord.py` 
This module is designed to handle the simultaneous recording of video streams from multiple cameras using threading. It captures video from four different streams and saves the recordings to disk. 
Key Features: 
- Initializes video capture objects for four streams.
- Sets frame size and frame rate for each capture object. 
- Creates `VideoWriter` objects for saving the recorded videos. - Uses threading to handle simultaneous recording from all four cameras. 
- Records for a specified duration or until a stop event is triggered. 
**Usage**: This script is used to record synchronized video streams from multiple cameras. 
**To-Do**: Timestamp videos for better synchronization

## Usage

1. Create YAML files with camera parameters for each camera (front, back, left, right).
2. Ensure these YAML files are placed in the `data/yaml` directory. 
3. Place video files in the `data/videos` directory and image files in the `data/images` directory.
4. Run `main.py --mode [mode]` with the appropriate mode (`video`, `imgvid`, `image`, or `stream`).


# Packages

To use the Python virtual environment, you should follow the following steps:

1. Open a terminal in the main directory.
2. Make sure virtualenv package is installed:
   ```bash
   pip install virtualenv
   ```
3. Run the following command to create a virtual environment named "venv":

   ```bash
    python3.11 -m venv venv
   ```

   This will create a new directory named "venv" that contains the virtual environment.

4. Activate the virtual environment by running the following command:
   pifa

   ```bash
   .\venv\Scripts\activate
   ```

   Once activated, your terminal prompt should change to indicate that you are now working within the virtual environment.

5. Install the list of required packages/dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

6. Instal pipreqs to automatically add packages that are being used to the requirements.txt:

   ```bash
   pip install pipreqs
   ```

7. Now to add all used packages to the requirements.txt do:

   ```bash
   pipreqs . --force
   ```

8. You can now write and run your Python code within the virtual environment.

Remember to deactivate the virtual environment when you're done by running the following command:

```bash
deactivate
```
