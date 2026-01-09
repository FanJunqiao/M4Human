# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022 wuzhiwyyx
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pymanopt.manifolds.group import SpecialOrthogonalGroup
from torch.nn import init
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import roi_align
from torchvision.ops.boxes import box_area

from .camera_parameters import *
from .transforms import project_3d_to_2d, transform_img_plane_to_cartesian


class RFTransform(GeneralizedRCNNTransform):
    """Data transformation. Normalize input data."""

    def __init__(self, min_size, max_size):
        super(RFTransform, self).__init__(min_size, max_size, [0] * 6, [1] * 6)

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        std, mean = torch.std_mean(image, dim=(1, 2), unbiased=True)
        return (image - mean[:, None, None]) / std[:, None, None]


def box_iou(boxes1, boxes2, eps=1e-7):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union


def _onnx_paste_mask_in_image(mask, box, im_h, im_w):
    """Copied from torchvision.models.detection.roi_heads"""

    one = torch.ones(1, dtype=torch.int64)
    zero = torch.zeros(1, dtype=torch.int64)

    w = box[2] - box[0] + one
    h = box[3] - box[1] + one
    w = torch.max(torch.cat((w, one)))
    h = torch.max(torch.cat((h, one)))

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, mask.size(0), mask.size(1)))

    # Resize mask
    mask = F.interpolate(mask, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
    x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
    y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
    y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

    unpaded_im_mask = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]

    # pad y
    zeros_y0 = torch.zeros(y_0, unpaded_im_mask.size(1))
    zeros_y1 = torch.zeros(im_h - y_1, unpaded_im_mask.size(1))
    concat_0 = torch.cat((zeros_y0, unpaded_im_mask.to(dtype=torch.float32), zeros_y1), 0)[0:im_h, :]
    # pad x
    zeros_x0 = torch.zeros(concat_0.size(0), x_0)
    zeros_x1 = torch.zeros(concat_0.size(0), im_w - x_1)
    im_mask = torch.cat((zeros_x0, concat_0, zeros_x1), 1)[:, :im_w]
    return im_mask


def project_3d_to_pixel(points, M, P, D, K):
    pts = torch.ones((1, points.shape[0]), device=points.device)
    pts = torch.cat([points.t(), pts], dim=0)
    cc = torch.mm(M, pts)
    pc = (cc[:2] / cc[2]).t()
    fx, fy, cx, cy, Tx, Ty = P[0, 0], P[1, 1], P[0, 2], P[1, 2], P[0, 3], P[1, 3]
    uv_rect_x, uv_rect_y = pc[:, 0], pc[:, 1]
    xp, yp = (uv_rect_x - cx - Tx) / fx, (uv_rect_y - cy - Ty) / fy
    r2 = xp * xp + yp * yp
    r4 = r2 * r2
    r6 = r4 * r2
    a1 = 2 * xp * yp
    k1, k2, p1, p2, k3 = D
    barrel = 1 + k1 * r2 + k2 * r4 + k3 * r6
    xpp = xp * barrel + p1 * a1 + p2 * (r2 + 2 * (xp * xp))
    ypp = yp * barrel + p1 * (r2 + 2 * (yp * yp)) + p2 * a1
    kfx, kcx, kfy, kcy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    u = xpp * kfx + kcx
    v = ypp * kfy + kcy
    pts = torch.stack([u, v], dim=0).t()
    pts = pts / 2
    return pts


class RadarToImgProjection(nn.Module):
    def __init__(self, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(RadarToImgProjection, self).__init__()
        self.dist_rot, self.dist_trans, self.deg_rot = 0, 0, 0
        self.point_dist_euclid = 0
        self.point_ang_rad = 0
        self.point_ang_deg = 0
        self.elem_rot_mean = 0
        self.elem_rot = 0
        self.rotation_mat = 0

        self.so3 = SpecialOrthogonalGroup(n=3)
        self.register_parameter(
            "rotation",
            nn.Parameter(torch.tensor(np.mean(R79.T, axis=1), **factory_kwargs)),
        )
        self.register_parameter("translation", nn.Parameter(torch.tensor(t79, **factory_kwargs)))
        self.register_parameter("fx", nn.Parameter(torch.tensor(fx, **factory_kwargs)))
        self.register_parameter("fy", nn.Parameter(torch.tensor(fy, **factory_kwargs)))
        self.register_parameter("ppx", nn.Parameter(torch.tensor(ppx, **factory_kwargs)))
        self.register_parameter("ppy", nn.Parameter(torch.tensor(ppy, **factory_kwargs)))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.rotation, a=math.sqrt(5))
        init.kaiming_uniform_(self.translation, a=math.sqrt(5))

    def exponential_map(self, omega):
        def skew_matrix(v):
            return torch.Tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]).to(v)

        assert omega.shape == (3,)
        t = torch.linalg.norm(omega)
        A_omega = skew_matrix(omega / t)
        return torch.eye(3).to(omega) + torch.sin(t) * A_omega + (1 - torch.cos(t)) * A_omega @ A_omega

    def calc_v_alignment(self, h_proposals, project):
        z_min, z_max = 64.6 / 2, 136.6 / 2
        v_props = []
        for prop in h_proposals:
            v = prop.clone()
            v[:, 0] = z_min
            v[:, 2] = z_max
            v_props.append(v)
        if not project:
            return v_props
        return v_props

    def forward(
        self,
        h_proposals,
        project=True,
        image_size=(224, 128),
        alignment=True,
        v_props=None,
        normed=False,
        masksize=(320, 240),
    ):
        if normed:
            proposals = torch.zeros(h_proposals.shape).to(h_proposals)
            proposals[:, :, [0, 2]] = h_proposals[:, :, [0, 2]] * image_size[1]
            proposals[:, :, [1, 3]] = h_proposals[:, :, [1, 3]] * image_size[0]
            h_proposals = proposals
            if v_props is not None:
                proposals = torch.zeros(v_props.shape).to(v_props)
                proposals[:, :, [0, 2]] = v_props[:, :, [0, 2]] * image_size[1]
                proposals[:, :, [1, 3]] = v_props[:, :, [1, 3]] * image_size[0]
                v_props = proposals
        if alignment:
            v_props = self.calc_v_alignment(h_proposals, project)

        proposals = []
        for b, (h, v) in enumerate(zip(h_proposals, v_props)):
            n_bbox = h.shape[0]
            """ convert 2D bboxes in hori and vert in image plane to Cartesian coordinate """
            cart_h_x0 = transform_img_plane_to_cartesian(
                x=h[:, 0],
                img_size=image_size[1],
                r_min=x_range[0],
                r_max=x_range[1],
                reverse=False,
            )
            cart_h_x1 = transform_img_plane_to_cartesian(
                x=h[:, 2],
                img_size=image_size[1],
                r_min=x_range[0],
                r_max=x_range[1],
                reverse=False,
            )
            cart_h_z0 = transform_img_plane_to_cartesian(
                x=h[:, 1],
                img_size=image_size[0],
                r_min=z_range[0],
                r_max=z_range[1],
                reverse=True,
            )
            cart_h_z1 = transform_img_plane_to_cartesian(
                x=h[:, 3],
                img_size=image_size[0],
                r_min=z_range[0],
                r_max=z_range[1],
                reverse=True,
            )
            cart_h = torch.cat(
                (
                    cart_h_x0[:, None],
                    cart_h_z0[:, None],
                    cart_h_x1[:, None],
                    cart_h_z1[:, None],
                ),
                dim=1,
            )
            cart_v_y0 = transform_img_plane_to_cartesian(
                x=v[:, 0],
                img_size=image_size[1],
                r_min=y_range[0],
                r_max=y_range[1],
                reverse=False,
            )
            cart_v_y1 = transform_img_plane_to_cartesian(
                x=v[:, 2],
                img_size=image_size[1],
                r_min=y_range[0],
                r_max=y_range[1],
                reverse=False,
            )
            cart_v_z0 = transform_img_plane_to_cartesian(
                x=v[:, 1],
                img_size=image_size[0],
                r_min=z_range[0],
                r_max=z_range[1],
                reverse=True,
            )
            cart_v_z1 = transform_img_plane_to_cartesian(
                x=v[:, 3],
                img_size=image_size[0],
                r_min=z_range[0],
                r_max=z_range[1],
                reverse=True,
            )
            cart_v = torch.cat(
                (
                    cart_v_y0[:, None],
                    cart_v_z0[:, None],
                    cart_v_y1[:, None],
                    cart_v_z1[:, None],
                ),
                dim=1,
            )

            """ calc 3D bbox in the radar plane with horizontal and vertical bboxes """
            x0, y0, z1, x1, y1, z0 = (
                cart_h[:, 0:1],
                cart_v[:, 0:1],
                cart_h[:, 1:2],
                cart_h[:, 2:3],
                cart_v[:, 2:3],
                cart_h[:, 3:4],
            )
            cart_3d_radar = torch.cat(
                (
                    x0,
                    y0,
                    z0,
                    x1,
                    y0,
                    z0,
                    x0,
                    y1,
                    z0,
                    x0,
                    y0,
                    z1,
                    x1,
                    y1,
                    z0,
                    x1,
                    y0,
                    z1,
                    x0,
                    y1,
                    z1,
                    x1,
                    y1,
                    z1,
                ),
                dim=1,
            )
            cart_3d_radar = torch.reshape(cart_3d_radar, (n_bbox, 8, 3)).permute((0, 2, 1))

            """ transformation from 3D Radar to 3D Camera with Rotation and Translation matrix """
            rotation = self.exponential_map(self.rotation)
            rot_mat = torch.linalg.inv(rotation).to(cart_3d_radar)[None]
            trans = -self.translation[None].to(cart_3d_radar)
            cart_3d_camera = (torch.matmul(rot_mat, cart_3d_radar + trans)).permute(0, 2, 1)

            """ projection from 3D camera coordinate to 2D camera image """
            image_2d_camera = project_3d_to_2d(cart_3d_camera, self.fx, self.fy, self.ppx, self.ppy)
            if n_bbox == 0:
                proposals.append(torch.zeros((0, 4)).to(image_2d_camera))
            else:
                image_2d_camera = (
                    torch.sort(image_2d_camera, dim=-1)[0][:, :, [0, -1]].permute(0, 2, 1).reshape(n_bbox, -1)
                )
                proposals.append(image_2d_camera / 2)
            if normed:
                proposals[b][:, [0, 2]] /= masksize[0]
                proposals[b][:, [1, 3]] /= masksize[1]

        return v_props, proposals


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)

    return roi_align(gt_masks, torch.clamp(rois, 0, 1000), (M, M), 1.0)[:, 0]


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels.long()],
        mask_targets,
    )
    return mask_loss


def iou(mask1, mask2):
    inter = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    return torch.sum(inter.flatten(1), dim=1) / torch.sum(union.flatten(1), dim=1)


def multiprod(A, B):
    if A.ndim == 2:
        return A @ B
    return torch.einsum("ijk,ikl->ijl", A, B)


def multitransp(A):
    if A.ndim == 2:
        return A.T
    return torch.permute(A, (0, 2, 1))
