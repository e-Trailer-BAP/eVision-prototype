"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fisheye Camera calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage:
    python calibration.py \
        -grid 9x7 \
        -out fisheye.yaml \
        -framestep 20 \
        --resolution 1280x960
        --fisheye
"""
# Change things to whatever is necessary

import argparse
import os
import numpy as np
import cv2

# Directory to save the camera parameter file
TARGET_DIR = os.path.join(os.getcwd(), "yaml")

# Default parameter file
DEFAULT_PARAM_FILE = os.path.join(TARGET_DIR, "camera_params.yaml")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()

    # Input video stream
    # parser.add_argument("-i", "--input", type=int, default=0,
    #                     help="input camera device")
    
    # Chessboard pattern size
    parser.add_argument("-grid", "--grid", default="7x9",
                        help="size of the calibrate grid pattern")
    
    # Camera resolution
    parser.add_argument("-r", "--resolution", default="1280x960",
                        help="resolution of the camera image")
    
    # Frame step for calibration
    parser.add_argument("-framestep", type=int, default=20,
                        help="use every nth frame in the video")
    
    # Output file for camera parameters
    parser.add_argument("-o", "--output", default=DEFAULT_PARAM_FILE,
                        help="path to output yaml file")
    # Flag to indicate if the camera is fisheye
    parser.add_argument("-fisheye", "--fisheye", action="store_true",
                        help="set true if this is a fisheye camera")
    
    # Flip method for the camera
    # parser.add_argument("-flip", "--flip", default=0, type=int,
    #                     help="flip method of the camera")
    
    # Flag to not use gstreamer for camera capture
    # parser.add_argument("--no_gst", action="store_true",
    #                     help="set true if not use gstreamer for the camera capture")
    
    args = parser.parse_args()

    if args.fisheye:
        print('Fisheye camera installed')

    # Create target directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    # Display text for user interaction
    # text1 = "press c to calibrate"
    text2 = "press q to quit"
    text3 = "device: webcam"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.6

    # Parse the resolution
    resolution_str = args.resolution.split("x")
    W = int(resolution_str[0])
    H = int(resolution_str[1])
    
    # Parse the grid size
    grid_size = tuple(int(x) for x in args.grid.split("x"))
    grid_points = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_points[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    # Lists to store object points and image points
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    # Open the OBS Virtual Webcam (usually at index 1, adjust if necessary)
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream from OBS Virtual Webcam")
        exit()

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Main loop for capturing frames and detecting chessboard corners
    quit = False
    do_calib = False
    i = -1

    while True:
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame is read correctly ret is True
        if not ret:
            print("Error: Failed to capture image")
            break

        img = frame

        if i % args.framestep != 0:
            continue

        if i == 20:
            print(img.shape)

        # # Display the resulting frame
        # cv2.imshow('OBS Virtual Webcam', frame)

        # # Break the loop on 'q' key press
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # # When everything is done, release the capture and close all windows
    # cap.release()
    # cv2.destroyAllWindows()

        if i == 20:
            print(img.shape)
        print("searching for chessboard corners in frame " + str(i) + "...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS
        )
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), term)
            print("OK")
            imgpoints.append(corners)
            objpoints.append(grid_points)
            # filename = f"C:/Users/Infer/Documents/TU Delft/BAP/e-Vision/data/calib_dataframe_{len(objpoints)}.jpg"  # Save frames in the specified directory
            # cv2.imwrite(filename, frame)
            # print(f"Saved frame {len(objpoints)}")

            cv2.drawChessboardCorners(img, grid_size, corners, found)
            #filename = f"C:/Users/Infer/Documents/Git/BAP/e-Vision/data/calib_data/frame_{len(objpoints)}.jpg"  # Save frames in the specified directory
            #cv2.imwrite(filename, frame)
            print(f"Saved frame {len(objpoints)}")

        # Display instructions on the frame
        # cv2.putText(img, text1, (20, 70), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text2, (20, 110), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text3, (20, 30), font, fontscale, (255, 200, 0), 2)
        cv2.imshow("corners", img)
        

        if len(objpoints) < 30:
            print("Less than 30 corners (%d) detected, calibration failed" % (len(objpoints)))
        else:
            print("\nPerforming calibration...\n")
            do_calib = True
            break

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            quit = True
            break

    # Clean up and exit if quit is requested
    if quit:
        cap.release()
        cv2.destroyAllWindows()

    # Perform calibration if enough points are collected
    if do_calib:
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             #cv2.fisheye.CALIB_CHECK_COND +
                             cv2.fisheye.CALIB_FIX_SKEW)

        # Use fisheye calibration if specified
        if args.fisheye:
            ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (W, H),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                (W, H),
                None,
                None
            )

        # Save calibration results
        if ret:
            fs = cv2.FileStorage(args.output, cv2.FILE_STORAGE_WRITE)
            fs.write("resolution", np.int32([W, H]))
            fs.write("camera_matrix", K)
            fs.write("dist_coeffs", D)
            fs.release()
            print("successfully saved camera data")

            # Read and print the saved yaml file
            with open(args.output, 'r') as f:
                yaml_content = f.read()
                print("Generated YAML file content:")
                print(yaml_content)

            cv2.putText(img, "Success!", (220, 240), font, 2, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Failed!", (220, 240), font, 2, (0, 0, 255), 2)

        # Display the result on the frame
        cv2.imshow("corners", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
