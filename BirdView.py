import cv2
import numpy as np
from Utilities import init_constants, mean_luminance_ratio, adjust_luminance, get_weight_mask_matrix, make_white_balance

class BirdView:
    def __init__(self):
        project_shapes = init_constants()
        # self.set_params(project_shapes)
        self.total_w = project_shapes["canvas_shape"]["front"][0]
        self.total_h = project_shapes["canvas_shape"]["left"][0]
        self.yt = project_shapes["canvas_shape"]["front"][1]
        self.yb = self.total_h - self.yt
        self.xl = project_shapes["canvas_shape"]["left"][1]
        self.xr = self.total_w - self.xl
        self.image = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
    
    def update_frames(self, new_frames):
        self.frames = new_frames

    def merge(self, imA, imB, k):
        if k < 0 or k >= len(self.weights):
            raise IndexError("Invalid index for weights.")

        G = self.weights[k]
        if G.shape != imA.shape:
            G = cv2.merge([G]*3)
        
        imA_double = imA.astype(np.float64)
        imB_double = imB.astype(np.float64)
        G_double = G.astype(np.float64)
        
        merged = imA_double * G_double + imB_double * (1.0 - G_double)
        merged = merged.astype(np.uint8)
        
        return merged

    def stitch_all_parts(self):
        front, back, left, right = self.frames
        # self.F(front[:self.yt, self.xl:self.xr])
        self.B(back[:self.yt, self.xl:self.xr])
        self.L(left[self.yt:self.yb, :self.xl])
        self.R(right[self.yt:self.yb, :self.xl])
        # self.FL(self.merge(front[:self.yt, :self.xl], left[:self.yt, :self.xl], 0))
        # self.FR(self.merge(front[:self.yt, self.xr:], right[:self.yt, :self.xl], 1))
        self.FL( left[:self.yt, :self.xl])
        self.FR( right[:self.yt, :self.xl])
        self.BL(self.merge(back[:self.yt, :self.xl], left[self.yb:, :self.xl], 2))
        self.BR(self.merge(back[:self.yt, self.xr:], right[self.yb:, :self.xl], 3))

    def load_car_image(self, data_path):
        car_image_path = f"{data_path}/images/car.png"
        self.car_image = cv2.imread(car_image_path)
        if self.car_image is not None:
            self.car_image = cv2.resize(self.car_image, (self.xr - self.xl, self.yb - self.yt))
        else:
            print(f"Error: Unable to load car image: {car_image_path}")

    def copy_car_image(self):
        self.C(self.car_image)
    
    # def set_params(self, project_shapes):
    #     self.total_w = project_shape["canvas_shape"]["front"][0]
    #     self.total_h = project_shapes["canvas_shape"]["left"][0]
    #     self.yt = project_shapes["front"][1]
    #     self.yb = self.total_h - self.yt
    #     self.xl = project_shapes["left"][1]
    #     self.xr = self.total_w - self.xl
    
    def make_luminance_balance(self):
        def tune(x):
            return x * np.exp((1 - x) * 0.5) if x >= 1 else x * np.exp((1 - x) * 0.8)

        front, back, left, right = self.frames

        a1 = mean_luminance_ratio(right[:self.yt, :, 0], front[:, self.xr:, 0], self.masks[1])
        a2 = mean_luminance_ratio(right[:self.yt, :, 1], front[:, self.xr:, 1], self.masks[1])
        a3 = mean_luminance_ratio(right[:self.yt, :, 2], front[:, self.xr:, 2], self.masks[1])

        b1 = mean_luminance_ratio(back[:, self.xr:, 0], right[self.yb:, :, 0], self.masks[3])
        b2 = mean_luminance_ratio(back[:, self.xr:, 1], right[self.yb:, :, 1], self.masks[3])
        b3 = mean_luminance_ratio(back[:, self.xr:, 2], right[self.yb:, :, 2], self.masks[3])

        c1 = mean_luminance_ratio(left[self.yb:, :, 0], back[:, :self.xl, 0], self.masks[2])
        c2 = mean_luminance_ratio(left[self.yb:, :, 1], back[:, :self.xl, 1], self.masks[2])
        c3 = mean_luminance_ratio(left[self.yb:, :, 2], back[:, :self.xl, 2], self.masks[2])
        
        d1 = mean_luminance_ratio(front[:, :self.xl, 0], left[:self.yt, :, 0], self.masks[0])
        d2 = mean_luminance_ratio(front[:, :self.xl, 1], left[:self.yt, :, 1], self.masks[0])
        d3 = mean_luminance_ratio(front[:, :self.xl, 2], left[:self.yt, :, 2], self.masks[0])

        t1 = np.power(a1 * b1 * c1 * d1, 0.25)
        t2 = np.power(a2 * b2 * c2 * d2, 0.25)
        t3 = np.power(a3 * b3 * c3 * d3, 0.25)

        x1 = tune(t1 / np.sqrt(d1 / a1))
        x2 = tune(t2 / np.sqrt(d2 / a2))
        x3 = tune(t3 / np.sqrt(d3 / a3))

        front[:, :, 0] = adjust_luminance(front[:, :, 0], x1)
        front[:, :, 1] = adjust_luminance(front[:, :, 1], x2)
        front[:, :, 2] = adjust_luminance(front[:, :, 2], x3)

        y1 = tune(t1 / np.sqrt(b1 / c1))
        y2 = tune(t2 / np.sqrt(b2 / c2))
        y3 = tune(t3 / np.sqrt(b3 / c3))

        back[:, :, 0] = adjust_luminance(back[:, :, 0], y1)
        back[:, :, 1] = adjust_luminance(back[:, :, 1], y2)
        back[:, :, 2] = adjust_luminance(back[:, :, 2], y3)

        z1 = tune(t1 / np.sqrt(c1 / d1))
        z2 = tune(t2 / np.sqrt(c2 / d2))
        z3 = tune(t3 / np.sqrt(c3 / d3))

        left[:, :, 0] = adjust_luminance(left[:, :, 0], z1)
        left[:, :, 1] = adjust_luminance(left[:, :, 1], z2)
        left[:, :, 2] = adjust_luminance(left[:, :, 2], z3)

        w1 = tune(t1 / np.sqrt(a1 / b1))
        w2 = tune(t2 / np.sqrt(a2 / b2))
        w3 = tune(t3 / np.sqrt(a3 / b3))

        right[:, :, 0] = adjust_luminance(right[:, :, 0], w1)
        right[:, :, 1] = adjust_luminance(right[:, :, 1], w2)
        right[:, :, 2] = adjust_luminance(right[:, :, 2], w3)

        self.frames[0] = front
        self.frames[1] = back
        self.frames[2] = left
        self.frames[3] = right

    def get_weights_and_masks(self, images):
        front, back, left, right = images
        G0, M0 = get_weight_mask_matrix(front[:, :self.xl], left[:self.yt, :])
        G1, M1 = get_weight_mask_matrix(front[:, self.xr:], right[:self.yt, :])
        G2, M2 = get_weight_mask_matrix(back[:, :self.xl], left[self.yb:, :])
        G3, M3 = get_weight_mask_matrix(back[:, self.xr:], right[self.yb:, :])

        self.weights = [G0, G1, G2, G3]
        self.masks = [M0 / 255.0, M1 / 255.0, M2 / 255.0, M3 / 255.0]

        return np.zeros((1, 4), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8)

    def make_white_balance(self):
        self.image = make_white_balance(self.image)

    def getImage(self):
        return self.image

    def F(self, region):
        self.image[:self.yt, self.xl:self.xr] = region

    def B(self, region):
        self.image[self.yb:, self.xl:self.xr] = region

    def L(self, region):
        self.image[self.yt:self.yb, :self.xl] = region

    def R(self, region):
        self.image[self.yt:self.yb, self.xr:] = region

    def FL(self, region):
        self.image[:self.yt, :self.xl] = region

    def FR(self, region):
        self.image[:self.yt, self.xr:] = region

    def BL(self, region):
        self.image[self.yb:, :self.xl] = region

    def BR(self, region):
        self.image[self.yb:, self.xr:] = region

    def C(self, region):
        self.image[self.yt:self.yb, self.xl:self.xr] = region
        
