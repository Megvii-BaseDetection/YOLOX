"""
Full torchscript model with custom business logic.

Input signature:
    {
        "inputs":
            {
                "input": torch.Tensor ((1,3,416,416), dtype: torch.float32)
                "zone": torch.Tensor (dtype: torch.float32)
            }
    }

Output signature:
    tuple(boxes: torch.Tensor, in_zone: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, boxes_centers: torch.Tensor)
"""

import torch
import torchvision
import torch.nn as nn


class DeployModule(nn.Module):
    def __init__(self, model: nn.Module, class_num: int, conf_thre: float, nms_thre: float):
        super().__init__()
        self.model = model
        self.class_num = class_num
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre

    @torch.jit.export
    def forward(self, x: dict[str, dict[str, torch.Tensor]], targets=None):
        image = x['inputs']['input'].permute(0, 3, 1, 2)
        zone = x['inputs']['zone']
        raw_output = self.model(image)
        raw_boxes = self.extract_raw_boxes(raw_output, self.class_num, self.conf_thre, self.nms_thre)

        boxes, scores, classes = self.extract_boxes_scores_and_classes(raw_boxes)
        in_zone = self._points_inside_zone(self._get_boxes_centers_xy(boxes), zone)
        boxes_centers = self._get_boxes_centers_yx(boxes)
        boxes = self._get_boxes_yxyx(boxes)
        return boxes, in_zone, scores, classes, boxes_centers

    @torch.jit.export
    def _get_boxes_centers_yx(self, boxes: torch.Tensor) -> torch.Tensor:
        x_coords = (boxes[:, 0] + boxes[:, 2]) / 2
        y_coords = (boxes[:, 1] + boxes[:, 3]) / 2
        return torch.stack([y_coords, x_coords], dim=1)

    @torch.jit.export
    def _get_boxes_centers_xy(self, boxes: torch.Tensor) -> torch.Tensor:
        centers = self._get_boxes_centers_yx(boxes)
        return torch.stack([centers[:, 1], centers[:, 0]], dim=1)

    @torch.jit.export
    def _get_boxes_yxyx(self, boxes: torch.Tensor) -> torch.Tensor:
        return torch.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], dim=1)

    @torch.jit.export
    def _points_inside_zone(self, points: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        points_x = points[:, 0].unsqueeze(1)
        points_y = points[:, 1].unsqueeze(1)
        zone_shifted = torch.roll(zone, shifts=1, dims=0)
        xi, yi = zone[:, 0], zone[:, 1]
        xj, yj = zone_shifted[:, 0], zone_shifted[:, 1]
        gyi = yi > points_y
        gyj = yj > points_y
        gx = (xj - xi) * (points_y - yi) / (yj - yi) + xi
        intersect_mask = (gyi != gyj) & (gx > points_x)
        result = torch.sum(intersect_mask, dim=1) % 2 > 0
        return result

    @torch.jit.export
    def extract_boxes_scores_and_classes(self, raw_boxes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_boxes = raw_boxes.to('cpu')
        boxes = raw_boxes[:, :4]
        scores, _ = torch.max(raw_boxes[:, 4:-1], dim=1)
        classes = raw_boxes[:, -1].to(torch.int32)
        return boxes, scores, classes

    @torch.jit.export
    def extract_raw_boxes(self, prediction) -> torch.Tensor:
        """
        Filter raw predictions. YoloX postprocess method including NMS method from torchvision.

        :param prediction: Raw prediction output from model.
        :return: Filtered predictions.
        """
        box_corner = torch.empty_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = []
        for i in range(prediction.shape[0]):
            image_pred = prediction[i]
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.class_num], 1, keepdim=True)
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.conf_thre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)

            if torch.nonzero(conf_mask).numel() > 0:
                conf_mask_indices = torch.nonzero(conf_mask).squeeze(1)
                detections = torch.index_select(detections, 0, conf_mask_indices)
                if not detections.size(0):
                    continue
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], self.nms_thre)

            detections = detections[nms_out_index]
            output.append(detections)
        output = torch.cat(output, dim=0)
        return output
