import cv2
import numpy as np


class AqStereo:
    def __init__(self, param_json=None):

        self.params = {
            "max_disp": 64,
            "min_disp": 0,
            "dark_val": 1,
            "window_size": 11,
            "texture_threshold": 5,
            "uniqueness_ratio": 5,
            "speckle_window_size": 41,
            "speckle_range": 8,
            "setDisp12MaxDiff": 1,
            "setPreFilterCap": 31,
            "im_width": 276,
            "im_height": 202,
        }

        # If a parameter exists in the param_json object then use that
        # otherwise use the default value
        if isinstance(param_json, dict):
            for v in self.params:
                if v in param_json:
                    self.params[v] = param_json[v]
                else:
                    print("warning! ", v, " not found in param_json")

        self._matcher = cv2.StereoBM_create(
            self.params["max_disp"], self.params["window_size"]
        )
        self._matcher.setROI1(None)
        self._matcher.setROI2(None)
        self._matcher.setPreFilterCap(self.params["setPreFilterCap"])
        self._matcher.setBlockSize(self.params["window_size"])
        self._matcher.setMinDisparity(self.params["min_disp"])
        self._matcher.setNumDisparities(self.params["max_disp"])
        self._matcher.setTextureThreshold(self.params["texture_threshold"])
        self._matcher.setUniquenessRatio(self.params["uniqueness_ratio"])
        self._matcher.setSpeckleWindowSize(self.params["speckle_window_size"])
        self._matcher.setSpeckleRange(self.params["speckle_range"])
        self._matcher.setDisp12MaxDiff(self.params["setDisp12MaxDiff"])

    def compute(self, l_im, r_im):
        im_size = (self.params["im_width"], self.params["im_height"])
        disparity = (
            self._matcher.compute(
                cv2.resize(l_im, im_size), cv2.resize(r_im, im_size)
            ).astype(np.float32)
            / 16.0
        )
        return disparity

    def infer_boxes_in_right(self, disparity, left_boxes, orig_shape):
        right_boxes = []
        disparity_boxes = []
        resize_scale_x = self.params["im_width"] / orig_shape[1]
        resize_scale_y = self.params["im_height"] / orig_shape[0]

        for b in left_boxes:
            x0 = b[0] * resize_scale_x
            x1 = b[2] * resize_scale_x
            y0 = b[1] * resize_scale_y
            y1 = b[3] * resize_scale_y

            x0 = max(0, x0)
            y0 = max(0, y0)

            if x1 >= self.params["im_width"]:
                x1 = self.params["im_width"] - 1
            if y1 >= self.params["im_height"]:
                y1 = self.params["im_height"] - 1

            bw = x1 - x0
            half_bw = bw * 0.5

            roi_l = disparity[int(y0) : int(y1), int(x0) : int(x1 - half_bw)]
            roi_r = disparity[int(y0) : int(y1), int(x0 + half_bw) : int(x1)]
            mask_l = roi_l >= 0.0
            mask_r = roi_r >= 0.0

            left_is_valid = np.count_nonzero(mask_l) > 0
            right_is_valid = np.count_nonzero(mask_r) > 0

            # get average disparity across the frame as a fallback
            valid_full = disparity[disparity >= 0.0]
            if np.count_nonzero(valid_full) == 0:
                mean_disp = 0.0
            else:
                mean_disp = valid_full.mean()

            width = b[2] - b[0]
            if not left_is_valid and not right_is_valid:
                # We have no info on how to translate the bounding boxes. Assume
                # average disparity across the frame as a default.
                disp_l, disp_r = mean_disp, mean_disp
                x0 = int((x0 - disp_l) / resize_scale_x)
                x1 = int((x1 - disp_r) / resize_scale_x)
            elif left_is_valid and not right_is_valid:
                # Invalid right disparity, use width of bbox.
                disp_l = np.mean(roi_l[mask_l])
                disp_r = 0
                x0 = int((x0 - disp_l) / resize_scale_x)
                x1 = int(max(x0 + width, x1))
            elif not left_is_valid and right_is_valid:
                # Invalid right disparity, use width of bbox.
                disp_l = 0
                disp_r = np.mean(roi_r[mask_r])
                x1 = int((x1 - disp_r) / resize_scale_x)
                x0 = int(min(x1 - width, x0))
            else:
                disp_l = np.mean(roi_l[mask_l])
                disp_r = np.mean(roi_r[mask_r])
                x0 = int((x0 - disp_l) / resize_scale_x)
                x1 = int((x1 - disp_r) / resize_scale_x)

            # ensure bbox fits within frame
            r_box = [
                max(x0, 0),
                max(b[1], 0),
                min(x1, orig_shape[1]),
                min(b[3], orig_shape[0]),
            ]

            right_boxes.append(r_box)
            disparity_boxes.append(0.5 * (disp_l + disp_r))

        return right_boxes, disparity_boxes
