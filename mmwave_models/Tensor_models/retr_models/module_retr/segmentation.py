# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) Facebook, Inc. and its affiliates.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
import io
from collections import defaultdict
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .box_ops import *
from .misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(
            (2, 2),
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.e1 = encoder_block(in_channels, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.b = conv_block(128, 128)
        self.d2 = decoder_block(128, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        self.outputs = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b = self.b(p3)
        d2 = self.d2(b, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs


class Generator(nn.Module):
    def __init__(self, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(channels, 64 * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class DETRsegm(nn.Module):
    def __init__(self, detr):
        super().__init__()
        self.detr = detr
        self.w = detr.w
        self.h = detr.h
        hidden_dim, nheads = 256, 4
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [64, 64, 64], 512)
        self.unet = Unet(32, 1)

        # modified for HMR
        # dim = 1024
        # dropout2 = 0.0
        # self.mesh_query_emb = nn.Parameter(torch.zeros(dim//hidden_dim, hidden_dim))
        
        
        dim = hidden_dim * 4
        dropout2 = 0.0
        self.mesh_query_emb = nn.Parameter(torch.zeros(1, hidden_dim))
        
        
        # point Prediction Head
        self.fc2_betas = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim//4),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(dim//4, 10)
        )
        self.fc2_pose_body = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim//4),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(dim//4, 63)
        )
        self.fc2_root_orient = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 3)
        )
        
        self.fc2_gender = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 1)
        )
        
        self.fc2_trans = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        self.register_buffer("trans_offset", torch.tensor([-3., 0.5, 0.0]))
        self.register_buffer("trans_scale", torch.tensor([6., 5.5, 3.0]))
        
        


    def forward(self, samples: NestedTensor, samples_ver: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            samples_ver = nested_tensor_from_tensor_list(samples_ver)

        # print(samples.tensors.shape, samples_ver.tensors.shape)


        features, pos = self.detr.backbone_h(samples)
        features_ver, pos_ver = self.detr.backbone_v(samples_ver)


        level = 0
        src, mask = features[level].decompose()
        src_ver, mask_ver = features_ver[level].decompose()

        bs, c, h, w = src.shape

        assert mask is not None
        src_proj = self.detr.input_proj(src)
        src_proj_ver = self.detr.input_proj_ver(src_ver)

        """ top-K selection for horizontal features """
        # print(src_proj.shape, src_proj_ver.shape, pos[level].shape, pos_ver[level].shape)
        # print(self.detr.topk)
        (
            topk_fea_hor,
            topk_pos_hor,
            topk_fea_ver,
            topk_pos_ver,
        ) = self.detr.topk_selection(src_proj, pos[level], src_proj_ver, pos_ver[level])
        # print(topk_fea_hor.shape, topk_pos_hor.shape, topk_fea_ver.shape, topk_pos_ver.shape)

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
        memory = self.detr.encoder(src_tokens, src_key_padding_mask=None, pos=pos_embed_fusion)
        query_embed = self.detr.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        mesh_query_embeded = self.mesh_query_emb.unsqueeze(1).repeat(1, bs, 1)
        mesh_tqt = torch.zeros_like(mesh_query_embeded)
        # print(self.mesh_query_emb.shape,mesh_query_embeded.shape, mesh_tqt.shape)
        mesh_hs, _ = self.detr.decoder(
            mesh_tqt,
            memory,
            memory_key_padding_mask=None,
            pos=pos_embed_fusion,
            query_pos=mesh_query_embeded,
        )
        # print(mesh_hs[-1].shape)


        x = mesh_hs[-1].squeeze()#.reshape(bs, -1)
        pred_betas = self.fc2_betas(x)
        pred_pose_body = self.fc2_pose_body(x)
        pred_root_orient = self.fc2_root_orient(x)

        gender_pred = torch.sigmoid(self.fc2_gender(x))
        pred_trans = self.fc2_trans(x) * self.trans_scale + self.trans_offset
        # print(pred_betas.shape, pred_pose_body.shape, pred_root_orient.shape, pred_trans.shape, gender_pred.shape)


        # hs, reference = self.detr.decoder(
        #     tgt,
        #     memory,
        #     memory_key_padding_mask=None,
        #     pos=pos_embed_fusion,
        #     query_pos=query_embed,
        # )
        # print(tgt.shape, hs.shape, reference.shape, pos_embed_fusion.shape)

        # """ detection head """
        # reference_before_sigmoid = inverse_sigmoid(reference)
        # print(reference_before_sigmoid.shape)
        # outputs_coords = []
        # for lvl in range(hs.shape[0]):
        #     print(lvl, self.detr.pos_dim)
        #     tmp = self.detr.bbox_embed(hs[lvl])
        #     print(tmp.shape)
        #     tmp[..., : self.detr.pos_dim] += reference_before_sigmoid
        #     outputs_coord = tmp.sigmoid()
        #     outputs_coords.append(outputs_coord)
        # outputs_coord = torch.stack(outputs_coords)

        # outputs_class = self.detr.class_embed(hs[-1])
        # print(outputs_class.shape)
        # pred_boxes = outputs_coord[-1]
        # out = {"pred_logits": outputs_class, "pred_boxes": pred_boxes}

        # """ normalized size bbox """
        # bbox = box_cxcyczwhd_to_xyzxyz(pred_boxes.reshape(-1, 6)).reshape(pred_boxes.shape)
        # hbox = torch.stack([bb[:, [0, 2, 3, 5]] for bb in bbox])  # x:azimuth, y:elevation, z:depth
        # vbox = torch.stack([bb[:, [1, 2, 4, 5]] for bb in bbox])
        # _, ibox = self.detr.calc_v_props(hbox, alignment=False, v_props=vbox, normed=True)
        # ibox = torch.stack(ibox)
        # out["pred_hboxes"] = box_xyxy_to_cxcywh(hbox)
        # out["pred_vboxes"] = box_xyxy_to_cxcywh(vbox)

        # iboxc = box_xyxy_to_cxcywh(ibox)
        # out["pred_iboxes"] = (
        #     torch.sigmoid(
        #         iboxc
        #         - self.detr.box_affine_transformer(
        #             torch.concatenate((iboxc, pred_boxes), dim=-1).view(-1, self.detr.num_queries)
        #         ).view(-1, self.detr.num_queries, 4)
        #     )
        #     + 1e-5
        # )

        # """ plane size bbox """
        # hbox_aug = box_cxcywh_to_xyxy(out["pred_hboxes"].detach().clone().reshape(-1, 4)).reshape(
        #     out["pred_hboxes"].shape
        # )
        # vbox_aug = box_cxcywh_to_xyxy(out["pred_vboxes"].detach().clone().reshape(-1, 4)).reshape(
        #     out["pred_vboxes"].shape
        # )
        # ibox_aug = box_cxcywh_to_xyxy(out["pred_iboxes"].detach().clone().reshape(-1, 4)).reshape(
        #     out["pred_iboxes"].shape
        # )
        # hbox_aug[:, :, [0, 2]], hbox_aug[:, :, [1, 3]] = (
        #     hbox_aug[:, :, [0, 2]] * self.detr.w,
        #     hbox_aug[:, :, [1, 3]] * self.detr.h,
        # )
        # vbox_aug[:, :, [0, 2]], vbox_aug[:, :, [1, 3]] = (
        #     vbox_aug[:, :, [0, 2]] * self.detr.w,
        #     vbox_aug[:, :, [1, 3]] * self.detr.h,
        # )
        # ibox_aug[:, :, [0, 2]], ibox_aug[:, :, [1, 3]] = (
        #     ibox_aug[:, :, [0, 2]] * self.detr.iw,
        #     ibox_aug[:, :, [1, 3]] * self.detr.ih,
        # )
        # out["pred_hboxes_aug"] = hbox_aug
        # out["pred_vboxes_aug"] = vbox_aug
        # out["proj_boxes"] = ibox_aug

        # if self.detr.aux_loss:
        #     out["aux_outputs"] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # bbox_mask = self.bbox_attention(hs[-1], src_proj_ver, mask=None)
        # mask_features = self.mask_head(
        #     src_proj,
        #     bbox_mask,
        #     [features_ver[2].tensors, features_ver[1].tensors, features_ver[0].tensors],
        # )

        # mask_features = F.interpolate(mask_features, size=(112, 112), mode="bilinear")
        # mask_features = self.unet(mask_features)

        # out["mask_logits"] = mask_features
        return pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred, None


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(4, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(4, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(4, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(4, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(4, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 32, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[-3], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[-2], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[-1], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def dice_loss(inputs, targets, num_boxes):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    def __init__(self, is_thing_map, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = (
            outputs["pred_logits"],
            outputs["pred_masks"],
            outputs["pred_boxes"],
        )
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)],
                        dtype=torch.bool,
                        device=keep.device,
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append(
                    {
                        "id": i,
                        "isthing": self.is_thing_map[cat],
                        "category_id": cat,
                        "area": a,
                    }
                )
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {
                    "png_string": out.getvalue(),
                    "segments_info": segments_info,
                }
            preds.append(predictions)
        return preds
