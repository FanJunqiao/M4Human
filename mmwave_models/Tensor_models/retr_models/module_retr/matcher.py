# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 Microsoft.
# Copyright (C) Facebook, Inc. and its affiliates.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcyczwhd_to_xyzxyz, box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, plane=None):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class MultiPlaneHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: list = [1, 1, 1, 1],
        cost_giou: list = [1, 1, 1, 1],
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_hbbox = cost_bbox[0]
        self.cost_vbbox = cost_bbox[1]
        self.cost_ibbox = cost_bbox[2]
        self.cost_3dbbox = cost_bbox[3]
        self.cost_hgiou = cost_giou[0]
        self.cost_vgiou = cost_giou[1]
        self.cost_igiou = cost_giou[2]
        self.cost_3dgiou = cost_giou[3]
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, planes=["h", "v"]):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        if "h" in planes:
            out_hbbox = outputs["pred_hboxes"].flatten(0, 1)
            tgt_hbbox = torch.cat([v["hboxes"] for v in targets])
        if "v" in planes:
            out_vbbox = outputs["pred_vboxes"].flatten(0, 1)
            tgt_vbbox = torch.cat([v["vboxes"] for v in targets])
        if "i" in planes:
            out_ibbox = outputs["pred_iboxes"].flatten(0, 1)
            tgt_ibbox = torch.cat([v["iboxes"] for v in targets])
        if "r3d" in planes:
            out_3dbbox = outputs["pred_boxes"].flatten(0, 1)
            tgt_3dbbox = torch.cat([v["3dboxes"] for v in targets])
        if "ori" in planes:
            out_bbox = outputs["pred_boxes"].flatten(0, 1)
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

        tgt_ids = torch.cat([v["labels"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        C = 0
        if "h" in planes:
            cost_hbbox = torch.cdist(out_hbbox, tgt_hbbox, p=1)
            cost_hgiou = -generalized_box_iou(box_cxcywh_to_xyxy(out_hbbox), box_cxcywh_to_xyxy(tgt_hbbox))
            C = C + self.cost_hbbox * cost_hbbox + self.cost_hgiou * cost_hgiou
        if "v" in planes:
            cost_vbbox = torch.cdist(out_vbbox, tgt_vbbox, p=1)
            cost_vgiou = -generalized_box_iou(box_cxcywh_to_xyxy(out_vbbox), box_cxcywh_to_xyxy(tgt_vbbox))
            C = C + self.cost_vbbox * cost_vbbox + self.cost_vgiou * cost_vgiou
        if "i" in planes:
            cost_ibbox = torch.cdist(out_ibbox, tgt_ibbox, p=1)
            cost_igiou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_ibbox), box_cxcywh_to_xyxy(tgt_ibbox)
            )  # nan is occurred
            C = C + self.cost_ibbox * cost_ibbox + self.cost_igiou * cost_igiou
        if "r3d" in planes:
            cost_3dbbox = torch.cdist(out_3dbbox, tgt_3dbbox, p=1)
            out_hbbox = out_3dbbox[:, [0, 2, 3, 5]]
            out_vbbox = out_3dbbox[:, [1, 2, 4, 5]]
            tgt_hbbox = tgt_3dbbox[:, [0, 2, 3, 5]]
            tgt_vbbox = tgt_3dbbox[:, [1, 2, 4, 5]]
            cost_hgiou_ = -generalized_box_iou(box_cxcywh_to_xyxy(out_hbbox), box_cxcywh_to_xyxy(tgt_hbbox))
            cost_vgiou_ = -generalized_box_iou(box_cxcywh_to_xyxy(out_vbbox), box_cxcywh_to_xyxy(tgt_vbbox))
            C = (
                C
                + self.cost_3dbbox * cost_3dbbox
                + (self.cost_3dgiou * cost_hgiou_ + self.cost_3dgiou * cost_vgiou_) / 2
            )
        if "ori" in planes:
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
            C = self.cost_hbbox * cost_bbox + self.cost_hgiou * cost_giou

        C = C + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["hboxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )
