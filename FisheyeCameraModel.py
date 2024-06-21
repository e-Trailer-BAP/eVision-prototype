import cv2
import yaml
import numpy as np
import os
import Utilities as utils

class FisheyeCameraModel:
    def __init__(self, camera_param_file, camera_name, debug=False, scale_xy=(1.0, 1.0), shift_xy=(0, 0)):
        if not os.path.exists(camera_param_file):
            raise RuntimeError("Cannot find camera param file")
        
        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.canvas_shape = self.init_constants()["canvas_shape"][self.camera_name]
        self.dXdY = self.init_constants()["dXdY"][self.camera_name]
        self.project_config = self.init_constants()["config"]
        self.load_camera_params()
        self.debug = debug
    
    def init_constants(self):
        return utils.init_constants()
    
    def read_matrix_simple(self, node):
        rows = node['rows']
        cols = node['cols']
        data = node['data']
        mat = np.array(data).reshape((rows, cols))
        return mat

    def read_point2f_simple(self, node):
        data = node['data']
        if len(data) != 2:
            raise RuntimeError("Expected a 2-element sequence for Point2f")
        return (data[0], data[1])

    def load_camera_params(self):
        with open(self.camera_file, 'r') as file:
            config = yaml.safe_load(file)
        
        try:
            self.camera_matrix = self.read_matrix_simple(config['camera_matrix'])
            self.dist_coeffs = self.read_matrix_simple(config['dist_coeffs'])
            self.resolution = self.read_matrix_simple(config['resolution']).reshape((1, 2))
            # self.scale_xy = self.read_point2f_simple(config['scale_xy'])
            # self.shift_xy = self.read_point2f_simple(config['shift_xy'])
            self.project_matrix = self.read_matrix_simple(config['project_matrix'])
            self.output_resolution = self.read_matrix_simple(config['output_resolution']).reshape((1, 2))[0]
            self.update_undistort_maps()
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            raise RuntimeError(f"Failed to load camera parameters from file: {self.camera_file}")

    def update_undistort_maps(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        
        width = int(self.resolution[0, 0])
        height = int(self.resolution[0, 1])
        
        self.undistort_map1, self.undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, np.eye(3), new_matrix, (width, height), cv2.CV_16SC2)

    def set_scale_and_shift(self, scale_xy, shift_xy):
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_maps()

    def undistort(self, image):
        return cv2.remap(image, self.undistort_map1, self.undistort_map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def project(self, image):
        return cv2.warpPerspective(image, self.project_matrix, self.output_resolution)

    def crop_and_flip(self, image):
        if self.debug:
            cv2.imwrite(f'data/images/precrop/{self.camera_name}.png', image)
        dX, dY = self.dXdY
        height, width = image.shape[:2]
        center_x = width // 2
        bottom_y = height

        x1 = int(max(0, center_x - dX))
        x2 = int(min(width, center_x + dX))
        y1 = int(max(0, bottom_y - dY))
        y2 = int(bottom_y)


        cropped_image = image[y1:y2, x1:x2]
        if self.debug:
            cv2.imwrite(f'data/images/postcrop/{self.camera_name}.png', cropped_image)

        if self.camera_name == "front":
            return cropped_image.copy()
        elif self.camera_name == "back":
            return cv2.flip(cropped_image, -1)
        elif self.camera_name == "left":
            transposed = cv2.transpose(cropped_image)
            return cv2.flip(transposed, 0)
        else:
            transposed = cv2.transpose(cropped_image)
            return cv2.flip(transposed, 1)

    def save_data(self):
        with cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE) as fs:
            fs.write("camera_matrix", self.camera_matrix)
            fs.write("dist_coeffs", self.dist_coeffs)
            fs.write("resolution", self.resolution)
            fs.write("project_matrix", self.project_matrix)
            fs.write("scale_xy", self.scale_xy)
            fs.write("shift_xy", self.shift_xy)
