# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 Microsoft.
# Copyright (C) Facebook, Inc. and its affiliates.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch.nn.functional as F
from torch import nn

from .backbone import build_backbone
from .box_ops import *
from .matcher import build_matcher
from .misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from .segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm, dice_loss, sigmoid_focal_loss


class DETR(nn.Module):
    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        num_classes,
        num_queries,
        aux_loss=False,
        topk=256,
        rw=128,
        rh=256,
        iw=240,
        ih=320,
    ):
        super().__init__()
        self.head_conv = 64
        self.w = rw
        self.h = rh
        self.iw, self.ih = iw, ih
        self.num_queries = num_queries
        self.calc_v_props = None
        self.dataname = None
        self.encoder = encoder
        self.decoder = decoder
        hidden_dim = 256
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_dim = 6
        self.pos_dim = 3
        self.bbox_embed = MLP(hidden_dim, hidden_dim, self.bbox_dim, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(64, hidden_dim, kernel_size=1)
        self.input_proj_ver = nn.Conv2d(64, hidden_dim, kernel_size=1)

        self.backbone_h = backbone[0]
        self.backbone_v = backbone[1]
        self.aux_loss = aux_loss
        self.topk = topk
        self.positions = {"hor": None, "ver": None}

        input_size = 10
        self.box_affine_transformer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, samples: NestedTensor, samples_ver: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            samples_ver = nested_tensor_from_tensor_list(samples_ver)

        features, pos = self.backbone(samples)
        features_ver, pos_ver = self.backbone(samples_ver)

        level = 0
        src, mask = features[level].decompose()
        src_ver, mask_ver = features_ver[level].decompose()

        bs, c, h, w = src.shape

        assert mask is not None
        src_proj = self.input_proj(src)
        src_proj_ver = self.input_proj_ver(src_ver)

        topk_fea_hor, topk_pos_hor, topk_fea_ver, topk_pos_ver = self.topk_selection(
            src_proj, pos[level], src_proj_ver, pos_ver[level]
        )

        """ Attention """
        src_tokens = torch.cat(
            [
                topk_fea_hor.flatten(2).permute(2, 0, 1),
                topk_fea_ver.flatten(2).permute(2, 0, 1),
            ]
        )
        pos_embed_fusion = torch.cat(
            [
                topk_pos_hor.flatten(2).permute(2, 0, 1),
                topk_pos_ver.flatten(2).permute(2, 0, 1),
            ],
            dim=0,
        )
        memory = self.encoder(src_tokens, src_key_padding_mask=None, pos=pos_embed_fusion)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs, reference = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_embed_fusion,
            query_pos=query_embed,
        )

        """ detection head """
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed(hs[lvl])
            tmp[..., : self.pos_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        pred_boxes = outputs_coord[-1]
        out = {"pred_logits": outputs_class[-1], "pred_boxes": pred_boxes}

        """ normalized size bbox """
        bbox = box_cxcyczwhd_to_xyzxyz(pred_boxes.reshape(-1, 6)).reshape(pred_boxes.shape)
        hbox = torch.stack([bb[:, [0, 2, 3, 5]] for bb in bbox])  # x:azimuth, y:elevation, z:depth
        vbox = torch.stack([bb[:, [1, 2, 4, 5]] for bb in bbox])  # x:azimuth, y:elevation, z:depth
        _, ibox = self.calc_v_props(hbox, alignment=False, v_props=vbox, normed=True)
        ibox = torch.stack(ibox)
        out["pred_hboxes"] = box_xyxy_to_cxcywh(hbox)
        out["pred_vboxes"] = box_xyxy_to_cxcywh(vbox)

        iboxc = box_xyxy_to_cxcywh(ibox)
        out["pred_iboxes"] = (
            torch.sigmoid(
                iboxc
                - self.box_affine_transformer(torch.concatenate((iboxc, pred_boxes), dim=-1).view(-1, 10)).view(
                    -1, self.num_queries, 4
                )
            )
            + 1e-5
        )

        """ plane size bbox """
        hbox_aug = box_cxcywh_to_xyxy(out["pred_hboxes"].detach().clone().reshape(-1, 4)).reshape(
            out["pred_hboxes"].shape
        )
        vbox_aug = box_cxcywh_to_xyxy(out["pred_vboxes"].detach().clone().reshape(-1, 4)).reshape(
            out["pred_vboxes"].shape
        )
        ibox_aug = box_cxcywh_to_xyxy(out["pred_iboxes"].detach().clone().reshape(-1, 4)).reshape(
            out["pred_iboxes"].shape
        )
        hbox_aug[:, :, [0, 2]], hbox_aug[:, :, [1, 3]] = (
            hbox_aug[:, :, [0, 2]] * self.w,
            hbox_aug[:, :, [1, 3]] * self.h,
        )
        vbox_aug[:, :, [0, 2]], vbox_aug[:, :, [1, 3]] = (
            vbox_aug[:, :, [0, 2]] * self.w,
            vbox_aug[:, :, [1, 3]] * self.h,
        )
        ibox_aug[:, :, [0, 2]], ibox_aug[:, :, [1, 3]] = (
            ibox_aug[:, :, [0, 2]] * self.iw,
            ibox_aug[:, :, [1, 3]] * self.ih,
        )
        out["pred_hboxes_aug"] = hbox_aug
        out["pred_vboxes_aug"] = vbox_aug
        out["proj_boxes"] = ibox_aug
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def topk_selection(self, src_proj, pos, src_proj_ver, pos_ver):
        topk_fea_hor, topk_pos_hor, index_hor = self.independent_topk_selection(src_proj, pos)
        topk_fea_ver, topk_pos_ver, index_ver = self.independent_topk_selection(src_proj_ver, pos_ver)
        self.positions["hor"] = index_hor
        self.positions["ver"] = index_ver
        return topk_fea_hor, topk_pos_hor, topk_fea_ver, topk_pos_ver

    def independent_topk_selection(self, src_proj, pos):
        bs, c, h, w = src_proj.shape
        sqrt_topk = int(np.sqrt(self.topk))
        l2_norm = torch.norm(src_proj, dim=1)
        v, i = torch.topk(l2_norm.view(bs, -1), dim=1, k=self.topk)
        topk_fetaures = src_proj.flatten(2, 3).permute(0, 2, 1)
        positional_encoding = pos.flatten(2, 3).permute(0, 2, 1)
        selected_features = []
        selected_features_pos = []
        for b_idx in range(topk_fetaures.shape[0]):
            selected_features.append(topk_fetaures[b_idx, i[b_idx]])
            selected_features_pos.append(positional_encoding[b_idx, i[b_idx]])
        selected_features = torch.stack(selected_features).permute(0, 2, 1).view(bs, 256, sqrt_topk, sqrt_topk)
        selected_features_pos = torch.stack(selected_features_pos).view(bs, 256, sqrt_topk, sqrt_topk)
        return selected_features, selected_features_pos, i


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, plane=["h", "v", "i"]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.plane = plane

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_hboxes(self, outputs, targets, indices, num_boxes):
        assert "pred_hboxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_hboxes"][idx]
        target_boxes = torch.cat([t["hboxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_hbbox"] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_hgiou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_vboxes(self, outputs, targets, indices, num_boxes):
        assert "pred_vboxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_vboxes"][idx]
        target_boxes = torch.cat([t["vboxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_vbbox"] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_vgiou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_iboxes(self, outputs, targets, indices, num_boxes):
        assert "pred_iboxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_iboxes"][idx]
        target_boxes = torch.cat([t["iboxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_ibbox"] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_igiou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_3dboxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["3dboxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_3dbbox"] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum()

        out_hbbox = src_boxes[:, [0, 2, 3, 5]]  # [0:c_azi, 1:c_ele, 2:c_depth, 3:w, 4:h, 5:d]
        out_vbbox = src_boxes[:, [1, 2, 4, 5]]
        tgt_hbbox = target_boxes[:, [0, 2, 3, 5]]
        tgt_vbbox = target_boxes[:, [1, 2, 4, 5]]
        cost_hgiou = generalized_box_iou(box_cxcywh_to_xyxy(out_hbbox), box_cxcywh_to_xyxy(tgt_hbbox))
        cost_vgiou = generalized_box_iou(box_cxcywh_to_xyxy(out_vbbox), box_cxcywh_to_xyxy(tgt_vbbox))
        cost = (cost_hgiou + cost_vgiou) / 2

        loss_giou = 1 - torch.diag(cost)
        losses["loss_3dgiou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "hboxes": self.loss_hboxes,
            "vboxes": self.loss_vboxes,
            "iboxes": self.loss_iboxes,
            "3dboxes": self.loss_3dboxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets, planes=self.plane)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            losses_list = ["labels", "3dboxes", "cardinality"]
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, planes=["r3d"])
                for loss in losses_list:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs = {"log": False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
