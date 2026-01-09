# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.roi_heads import expand_boxes, expand_masks, maskrcnn_inference
from smplx import SMPLXLayer


from .module_retr import box_ops
from .module_retr.detr import SetCriterion
from .module_retr.hubconf import detr_resnet18
from .module_retr.matcher import MultiPlaneHungarianMatcher
from .module_retr.segmentation import DETRsegm
from .module_retr.utils import RadarToImgProjection, RFTransform, maskrcnn_loss

def batch_rodrigues(theta):
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues' formula.

    Args:
        theta: [B, 3] axis-angle vectors (rotation vector)
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    angle = torch.norm(theta, dim=1, keepdim=True)  # [B, 1]
    axis = theta / (angle + 1e-8)                   # [B, 3], unit rotation axis

    # Create skew-symmetric cross-product matrix K for each axis
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=1).reshape(-1, 3, 3)  # [B, 3, 3]

    I = torch.eye(3, device=theta.device).unsqueeze(0)  # [1, 3, 3]
    angle = angle.unsqueeze(-1)  # [B, 1, 1]
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    R = I + sin * K + (1 - cos) * torch.bmm(K, K)  # [B, 3, 3]
    return R


class RETR(nn.Module):
    def __init__(
        self,
        smplx_model_paths=None,
        task: str = "DET",
        num_classes: int = 2,
        hidden_dim: int = 256,
        in_channels: int = [4,4],
        num_queries: int = 10,
        ratio: float = 0.6,
        topk: int = 256,
        thresh_mask: float = 0.5,
        path: str = None,
        rw: int = 128,
        rh: int = 256,
        iw: int = 320,
        ih: int = 240,
        loss_ce: float = 1.0,
        loss_hbbox: float = 0.5,
        loss_vbbox: float = 0.5,
        loss_ibbox: float = 1.0,
        loss_giou: float = 1.0,
        return_intermediate_dec: bool = True,
        **kwargs
    ):
        super().__init__()
        self.w, self.h = rw, rh
        self.iw, self.ih = iw, ih
        self.thresh_mask = thresh_mask
        self.transform = RFTransform(self.h, self.w)
        self.bbox_dim = 6
        self.model, self.postprocessor = detr_resnet18(
            num_classes=num_classes,
            return_postprocessor=True,
            num_queries=num_queries,
            ratio=ratio,
            return_intermediate_dec=return_intermediate_dec,
            topk=topk,
            rw=rw,
            rh=rh,
            iw=iw,
            ih=ih,
            **kwargs
        )

        self.model.backbone_h[0].body.body.conv1 = nn.Conv2d(
            in_channels[0], 64, kernel_size=4, stride=2, padding=3, bias=False
        )
        self.model.backbone_v[0].body.body.conv1 = nn.Conv2d(
            in_channels[1], 64, kernel_size=4, stride=2, padding=3, bias=False
        )

        self.in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=num_classes)
        self.model.num_queries = num_queries
        self.model.query_embed = nn.Embedding(num_queries, self.in_features)
        self.model.input_proj = nn.Conv2d(64, hidden_dim, kernel_size=1)
        self.model.input_proj_ver = nn.Conv2d(64, hidden_dim, kernel_size=1)

        """ For loss calculation """
        self.planes = ["h", "v", "i"]
        self.matcher = MultiPlaneHungarianMatcher()
        self.weight_dict = {
            "loss_ce": loss_ce,
            "loss_hbbox": loss_hbbox,
            "loss_vbbox": loss_vbbox,
            "loss_ibbox": loss_ibbox,
            "loss_hgiou": loss_giou,
            "loss_vgiou": loss_giou,
            "loss_igiou": loss_giou,
        }
        self.losses = ["labels", "hboxes", "iboxes", "vboxes", "cardinality"]
        self.criterion = SetCriterion(
            1,
            self.matcher,
            self.weight_dict,
            eos_coef=0.5,
            losses=self.losses,
            plane=self.planes,
        )

        self.model.calc_v_props = RadarToImgProjection()

        """ For segmentation """
        self.seg = True if task == "SEG" else False
        if self.seg:
            self.mask = True
            if path is not None:
                params = torch.load(path, map_location="cpu")
                model_dict = self.model.state_dict()
                for name, param in params.items():
                    if name.replace("model.", "") in model_dict:
                        model_dict[name.replace("model.", "")] = param

                self.model.load_state_dict(model_dict)
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model = DETRsegm(self.model)

        '''For SMPLX'''

        self.smplx_layer_male = SMPLXLayer(model_path=smplx_model_paths['male'], gender='male', use_pca=False)
        self.smplx_layer_female = SMPLXLayer(model_path=smplx_model_paths['female'], gender='female', use_pca=False)
        self.smplx_layer_neutral = SMPLXLayer(model_path=smplx_model_paths['neutral'], gender='neutral', use_pca=False)

        self.faces_male = self.smplx_layer_male.faces
        self.faces_female = self.smplx_layer_female.faces
        self.faces_neutral = self.smplx_layer_neutral.faces

        self.loss_weights = {
            'betas': 2.0,
            'pose_body': 5.0,
            'root_orient': 1.0,
            'trans': 1.0,
            'vertices': 0.001,
            'gender': 1.0
        }

    def paste_mask_in_image(self, mask, box, im_h, im_w):
        TO_REMOVE = 1
        w = int(box[2] - box[0] + TO_REMOVE)
        h = int(box[3] - box[1] + TO_REMOVE)
        if w > 2 * im_w or h > 2 * im_h:
            return torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)

        w = max(w, 1) if w < im_w else im_w
        h = max(h, 1) if h < im_h else im_h

        mask = mask.expand((1, 1, -1, -1))

        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = mask[0][0]

        im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
        x_0 = max(box[0], 0)
        x_1 = min(box[2] + 1, im_w)
        y_0 = max(box[1], 0)
        y_1 = min(box[3] + 1, im_h)
        _ = mask[(y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])]
        if im_mask[y_0:y_1, x_0:x_1].numel() == _.numel():
            im_mask[y_0:y_1, x_0:x_1] = _
        return im_mask

    def paste_masks_in_image(self, masks, boxes, img_shape, padding=1):
        masks, scale = expand_masks(masks, padding=padding)
        boxes = expand_boxes(boxes, scale).to(dtype=torch.int64)
        im_h, im_w = img_shape

        res = [self.paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]
        if len(res) > 0:
            ret = torch.stack(res, dim=0)[:, None]
        else:
            ret = masks.new_empty((0, 1, im_h, im_w))
        return ret

    def forward(self, tensor, targets=None):
        """
        RETR main process
        :param hor: [batch, n_frames, h, w] horizontal map
        :param ver: [batch, n_frames, h, w] vertical map
        :param targets: list of dictionaries with target data
        :return: preds, loss_dict (if targets are provided)
        """
        # tensor: [B, n_frames, H, W, D]
        # hor: [B, n_frames*D, H, W], ver: [B, n_frames*H, W, D]
        B, T, H, W, D = tensor.shape
        hor = tensor.permute(0, 1, 4, 2, 3).reshape(B, T * D, H, W)
        ver = tensor.permute(0, 1, 3, 4, 2).reshape(B, T * H, W, D)


        out = self.model(hor, ver)
        return out

    def get_smplx_output(self, pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender):
        vertices_list = []
        for i in range(gender.shape[0]):
            gender_value = gender[i].item()
            if gender_value > 0.5:
                smplx_layer = self.smplx_layer_male
            else:
                smplx_layer = self.smplx_layer_female

            pred_root_orient_matrix = batch_rodrigues(pred_root_orient[i].unsqueeze(0)).unsqueeze(0)
            pred_pose_body_matrix = batch_rodrigues(pred_pose_body[i].view(-1, 3)).view(-1, 21, 3, 3)

            output = smplx_layer(
                betas=pred_betas[i].unsqueeze(0),
                body_pose=pred_pose_body_matrix,
                global_orient=pred_root_orient_matrix,
                transl=pred_trans[i].unsqueeze(0)
            )
            vertices_list.append(output.vertices)
        vertices = torch.cat(vertices_list, dim=0)
        return vertices*1000

        # if targets is not None:
        #     hbox = torch.stack([tgt["hboxes"] for tgt in targets])
        #     vbox = torch.stack([tgt["vboxes"] for tgt in targets])
        #     ibox = torch.stack([tgt["iboxes"] for tgt in targets])
        #     masks = torch.stack([tgt["masks"] for tgt in targets])
        #     n_sbjs = torch.tensor([tgt["n_sbj"] for tgt in targets])

        #     target = []
        #     _, _, hm, wm = masks.shape
        #     for hbbox, vbbox, ibox, mask, n_sbj in zip(hbox, vbox, ibox, masks, n_sbjs):
        #         hbbox = box_ops.box_xyxy_to_cxcywh(hbbox[:n_sbj]).clone()
        #         vbbox = box_ops.box_xyxy_to_cxcywh(vbbox[:n_sbj]).clone()
        #         ibox = box_ops.box_xyxy_to_cxcywh(ibox[:n_sbj, :4]).clone()

        #         # Normalize coordinates
        #         hbbox[:, [0, 2]] /= self.w
        #         hbbox[:, [1, 3]] /= self.h
        #         vbbox[:, [0, 2]] /= self.w
        #         vbbox[:, [1, 3]] /= self.h
        #         ibox[:, [0, 2]] /= wm
        #         ibox[:, [1, 3]] /= hm

        #         new_3dbbox = torch.stack(
        #             [
        #                 hbbox[:, 0],
        #                 vbbox[:, 0],
        #                 hbbox[:, 1],
        #                 hbbox[:, 2],
        #                 vbbox[:, 2],
        #                 hbbox[:, 3],
        #             ],
        #             dim=1,
        #         )
        #         labels_bb = torch.zeros(len(hbbox), device=hbbox.device, dtype=torch.long)
        #         target.append(
        #             {
        #                 "boxes": hbbox.float(),
        #                 "hboxes": hbbox.float(),
        #                 "vboxes": vbbox.float(),
        #                 "iboxes": ibox.float(),
        #                 "3dboxes": new_3dbbox.float(),
        #                 "labels": labels_bb,
        #                 "masks": mask,
        #             }
        #         )

        # if self.seg:
        #     num_queries = self.model.detr.num_queries
        #     batch_size = len(out["pred_boxes"])
        #     device = hor.device

        #     gt_labels = [torch.zeros(num_queries, device=device) for _ in range(batch_size)]
        #     matched_ids = [l.clone().long() for l in gt_labels]

        #     if targets is not None:
        #         gt_masks = torch.stack([tgt["masks"].sum(0) for tgt in targets])
        #         gt_masks = (gt_masks > 0).long().unsqueeze(1)
        #         loss_seg = maskrcnn_loss(
        #             out["mask_logits"],
        #             out["proj_boxes"],
        #             gt_masks,
        #             gt_labels,
        #             matched_ids,
        #         )

        #     masks_probs = maskrcnn_inference(out["mask_logits"], [l.long() for l in gt_labels])

        #     predictions, final_mask_logits, final_instance_masks = [], [], []
        #     for pred_masks, boxes, logits in zip(masks_probs, out["proj_boxes"], out["pred_logits"]):
        #         best_indices = logits.sigmoid()[:, 0] > self.thresh_mask
        #         pred_masks = pred_masks[best_indices]
        #         boxes = boxes[best_indices]
        #         pred_masks = self.paste_masks_in_image(pred_masks, boxes, (self.ih, self.iw))

        #         h, w = pred_masks.shape[-2:]
        #         final_mask = torch.zeros(h, w, device=pred_masks.device)
        #         final_masks = torch.zeros(len(pred_masks), h, w, device=pred_masks.device)
        #         final_logits = torch.zeros(h, w, device=pred_masks.device)

        #         for q, current_mask in enumerate(pred_masks.detach()):
        #             mask_thresh = current_mask[0] > self.thresh_mask
        #             final_masks[q][mask_thresh] = 1
        #             final_mask[mask_thresh] = 1
        #             positive = current_mask[0] > 0
        #             final_logits[positive] = current_mask[0][positive]

        #         predictions.append(final_mask)
        #         final_mask_logits.append(final_logits)
        #         final_instance_masks.append(final_masks)

        #     out["masks"] = torch.stack(predictions)
        #     out["masks_logits"] = torch.stack(final_mask_logits)
        #     out["masks_inst"] = final_instance_masks

        # if targets is not None:
        #     loss_dict = self.criterion(out, target)
        #     if self.seg:
        #         loss_dict["loss_seg"] = loss_seg

        # preds = []
        # for image_boxes, bb_scores, hbboxes, vbboxes in zip(
        #     out["proj_boxes"],
        #     out["pred_logits"].sigmoid()[:, :, 0],
        #     out["pred_hboxes_aug"],
        #     out["pred_vboxes_aug"],
        # ):
        #     best_indices = bb_scores > self.thresh_mask
        #     preds.append(
        #         {
        #             "iboxes": image_boxes[best_indices],
        #             "scores": bb_scores[best_indices],
        #             "labels": torch.zeros(best_indices.sum(), device=bb_scores.device, dtype=torch.long),
        #             "hboxes": hbboxes[best_indices],
        #             "vboxes": vbboxes[best_indices],
        #         }
        #     )

        # if self.seg:
        #     if targets is not None:
        #         loss_dict["loss_seg"] = loss_seg
        #     for pred, fmask, ins_mask in zip(preds, out["masks"], out["masks_inst"]):
        #         pred["masks"] = fmask
        #         pred["masks_person"] = ins_mask

        # return (preds, loss_dict) if targets is not None else preds
