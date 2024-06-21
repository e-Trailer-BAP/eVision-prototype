from FisheyeCameraModel import FisheyeCameraModel
import os
import Utilities as utils
import cv2
import numpy as np
import math

def process_image(images, data_path, output_path, camera_models, prefix):
    projected = []
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not open or find the image: {image_path}")
        img = camera_models[i].undistort(img)
        img = camera_models[i].project(img)
        # img = camera_models[i].crop_and_flip(projected)
        # projected.append(img)

        # print("Bird's eye view stitched image after added white balance")
        cv2.imshow("image", img)
        cv2.imwrite(os.path.join(output_path, f'{prefix}-test.png'), img)
        cv2.waitKey(0)
        # cv2.imwrite(os.path.join(output_path, f'{prefix}-BEV.png'), birdview.getImage())

def find_centroids(Undistorted, HRL=30, SRL=160, VRL=20, HRU=12, SRU=40, VRU=40):
    # Load the image
        
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(Undistorted, cv2.COLOR_BGR2HSV)
    H=150
    S=220
    V=220
    lower_pink=np.array([H-HRL, S-SRL, V-VRL])
    upper_pink=np.array([H+HRU, S+SRU, V+VRU])
    
    # Create a mask for pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has at least 3 points
        if len(approx) >= 3:
            # Draw the contour in orange (for visualization purposes)
            cv2.drawContours(Undistorted, [approx], -1, (0, 255, 0), 2)
            
            # Calculate the centroid of the polygon
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
                # print("Centroid:", cX, cY)
                # Draw the centroid
                cv2.circle(Undistorted, (cX, cY), 5, (0, 255, 0), -1)  # Green for centroids
    
    # Display the image with the contours and centroids marked
    # cv2.imshow("Image with Contours and Centroids", Undistorted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return centroids, Undistorted

def order_centroids(centroids):
    # Sort centroids based on x-coordinate (ascending) to separate left and right
    print(centroids)
    centroids_sorted_by_x = sorted(centroids, key=lambda x: x[0])

    # Split into left and right halves
    left_half = centroids_sorted_by_x[:2]
    right_half = centroids_sorted_by_x[2:]

    # Sort left half by y-coordinate to get bottom left and top left
    top_left, bottom_left  = sorted(left_half, key=lambda x: x[1])

    # Sort right half by y-coordinate to get bottom right and top right
    top_right, bottom_right  = sorted(right_half, key=lambda x: x[1])

    # print(f"Bottom right: {bottom_right} Bottom left: {bottom_left} Top left: {top_left} Top right: {top_right}")
    return bottom_right, bottom_left, top_left, top_right

def project(image, camera_name, centroid):
    camx=500
    camy=500
    scale=1

    # Specify the width and the height of transformed image
    width, height = math.floor(scale*camx*2), math.floor(scale*camy)

    if camera_name == "front":
        offset=110
        br=[50,40+offset]
        bl=[-120,50+offset]
        tl=[-70,190+offset]
        tr=[100,160+offset]
    # print(centroid)
    pts1 = np.float32(centroid)
    pts2 = np.float32([[scale*(camx+br[0]), scale*(camy-br[1])],
                    [scale*(camx+bl[0]), scale*(camy-bl[1])],
                    [scale*(camx+tl[0]), scale*(camy-tl[1])],
                    [scale*(camx+tr[0]), scale*(camy-tr[1])]])
    #bottom right, bottom left, top left, top right

    # Apply perspective transform Method
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (width, height))

    # print(matrix)

    # # Show the result
    # print('pre transform')

    # cv2.imshow("pre",image)

    # print('post transform')
    # cv2.imshow("post",result)
    cv2.imwrite("data/output/testssdfs.png", result)
    # cv2.waitKey()
    return matrix


if __name__ == "__main__":
    mode = "image"
    camera_names = ["front"]
    # "back", "left", "right"]
    yamls = []
    calib_images = []
    camera_models = []
    images = []
    videos = []
    data_path = "data"
    calib_path = "data/calib_images"
    output_path = "data/output"
    prefix = "t5"
    debug=True
    scale_xy=(0.8, 0.8)
    shift_xy=(0, 0)

    if debug:
        print("Debug mode enabled")
        print(utils.init_constants()["config"])

    for name in camera_names:
        yamls.append(os.path.join(data_path, "yaml", f"{name}.yaml"))
        calib_images.append(os.path.join(data_path, "calib_images", f"{name}.png"))

        images.append(os.path.join(data_path, "images", f"{name}-5.png"))

    calib_models = [FisheyeCameraModel(yaml_file, name, debug, scale_xy, shift_xy) for yaml_file, name in zip(yamls, camera_names)]

    for calib_model in calib_models:
        image_path = os.path.join(calib_path, f"{calib_model.camera_name}.png")
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not open or find the image: {image_path}")
        undistorted = calib_model.undistort(img)
        centroid, detected = find_centroids(undistorted)

        if len(centroid) == 4:
            centroid = order_centroids(centroid)
            # print(centroid)
        else:
            print("Error: Unable to detect 4 centroids")
            print("Centroids detected:", len(centroid))
        projection_matrix = project(img, calib_model.camera_name, centroid)
        # cv2.imshow("Undistorted", undistorted)
        # cv2.imshow("Detected", detected)
        camera_model = FisheyeCameraModel(calib_model.camera_file, calib_model.camera_name, debug, scale_xy, shift_xy)
        camera_model.undistorted = undistorted
        print(projection_matrix)
        camera_model.project_matrix = projection_matrix
        camera_models.append(camera_model)

    # selected_camera_model = camera_models[0]

    process_image(images, data_path, output_path, camera_models, prefix)

