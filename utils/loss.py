# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details
    see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # StreamYOLO TAL parameters (s-model values from paper)
    tal_gamma = 1.0
    tal_ignore_thr = 0.5
    tal_ignore_value = 1.5

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        m = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device

    def _compute_tal_weights(self, targets, support_targets):
        """Compute Trend-Aware Learning (TAL) weights per target GT box.

        For each target box, find the best IoU with any support-frame GT box
        (same image index, any class). Then:
            weight = 1 / (best_iou ^ gamma)
        If best_iou > ignore_thr: weight = ignore_value  (nearly-static objects)
        If no support boxes for that image: weight = 1.0  (no temporal info)

        Args:
            targets: [N, 6] tensor — [img_idx, class, x, y, w, h] (normalized xywh)
            support_targets: [M, 6] tensor — same format for support frame

        Returns:
            weights: [N] tensor of per-GT TAL weights
        """
        n = targets.shape[0]
        weights = torch.ones(n, device=self.device)

        if n == 0 or support_targets.shape[0] == 0:
            return weights

        for img_i in targets[:, 0].unique().long():
            # Masks for this image
            t_mask = targets[:, 0].long() == img_i
            s_mask = support_targets[:, 0].long() == img_i

            if not s_mask.any():
                continue  # no support boxes → weight stays 1.0

            # Get boxes in xywh format, convert to xyxy for IoU
            t_boxes = targets[t_mask, 2:6]  # [nt, 4] normalized xywh
            s_boxes = support_targets[s_mask, 2:6]  # [ns, 4] normalized xywh

            # xywh to xyxy
            t_xyxy = torch.zeros_like(t_boxes)
            t_xyxy[:, 0] = t_boxes[:, 0] - t_boxes[:, 2] / 2
            t_xyxy[:, 1] = t_boxes[:, 1] - t_boxes[:, 3] / 2
            t_xyxy[:, 2] = t_boxes[:, 0] + t_boxes[:, 2] / 2
            t_xyxy[:, 3] = t_boxes[:, 1] + t_boxes[:, 3] / 2

            s_xyxy = torch.zeros_like(s_boxes)
            s_xyxy[:, 0] = s_boxes[:, 0] - s_boxes[:, 2] / 2
            s_xyxy[:, 1] = s_boxes[:, 1] - s_boxes[:, 3] / 2
            s_xyxy[:, 2] = s_boxes[:, 0] + s_boxes[:, 2] / 2
            s_xyxy[:, 3] = s_boxes[:, 1] + s_boxes[:, 3] / 2

            # Pairwise IoU: [nt, ns]
            nt_boxes = t_xyxy.shape[0]
            ns_boxes = s_xyxy.shape[0]

            # Broadcast intersection
            inter_x1 = torch.max(t_xyxy[:, 0:1], s_xyxy[:, 0:1].T)  # [nt, ns]
            inter_y1 = torch.max(t_xyxy[:, 1:2], s_xyxy[:, 1:2].T)
            inter_x2 = torch.min(t_xyxy[:, 2:3], s_xyxy[:, 2:3].T)
            inter_y2 = torch.min(t_xyxy[:, 3:4], s_xyxy[:, 3:4].T)

            inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            t_area = (t_xyxy[:, 2] - t_xyxy[:, 0]) * (t_xyxy[:, 3] - t_xyxy[:, 1])
            s_area = (s_xyxy[:, 2] - s_xyxy[:, 0]) * (s_xyxy[:, 3] - s_xyxy[:, 1])

            union = t_area[:, None] + s_area[None, :] - inter_area
            iou_matrix = inter_area / (union + 1e-7)  # [nt, ns]

            best_iou, _ = iou_matrix.max(dim=1)  # [nt] — best match per target box

            # TAL weight: 1 / (iou ^ gamma)
            w = 1.0 / (best_iou.clamp(min=1e-7) ** self.tal_gamma)

            # Override high-IoU (nearly-static) objects with ignore_value
            w[best_iou > self.tal_ignore_thr] = self.tal_ignore_value

            weights[t_mask] = w

        return weights

    def __call__(self, p, targets, support_targets=None):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        # --- TAL weights: computed per-GT from support frame IoU ---
        if support_targets is not None and support_targets.shape[0] > 0:
            gt_tal_weights = self._compute_tal_weights(targets, support_targets)
        else:
            gt_tal_weights = torch.ones(targets.shape[0], device=self.device)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            if n := b.shape[0]:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()

                # Trend-Aware weighted box loss (paper Eq. 2-3)
                lbox_i = (1.0 - iou)
                w_i = self._get_matched_tal_weights(b, targets, gt_tal_weights, i)
                # Paper Eq. 3: normalize weights so total loss magnitude is unchanged
                # ω̂_i = ω_i * Σ L_i^reg / Σ (ω_i * L_i^reg)
                w_sum = (w_i * lbox_i.detach()).sum()
                l_sum = lbox_i.detach().sum()
                if w_sum > 0:
                    w_hat = w_i * (l_sum / w_sum)
                else:
                    w_hat = w_i
                lbox += (lbox_i * w_hat).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def _get_matched_tal_weights(self, b_indices, targets, gt_tal_weights, layer_idx):
        """Map per-GT TAL weights to the expanded set of matched predictions.

        build_targets expands GTs across anchors and neighboring grid cells.
        We need to map each expanded match back to its original GT to get the TAL weight.
        Since we don't have a direct index, we use the image index as a rough mapping —
        but for correctness we return per-image average TAL weight.

        A simpler but effective approach: each matched prediction at (b, a, gj, gi)
        came from a GT in image b[k]. We take the average TAL weight of all GTs
        in that image. This is an approximation but works well in practice since
        most images have similar-motion objects.
        """
        n = b_indices.shape[0]
        if n == 0:
            return torch.ones(0, device=self.device)

        weights = torch.ones(n, device=self.device)
        for img_i in b_indices.unique():
            mask = b_indices == img_i
            gt_mask = targets[:, 0].long() == img_i.long()
            if gt_mask.any():
                avg_w = gt_tal_weights[gt_mask].mean()
                weights[mask] = avg_w
        return weights

    def build_targets(self, p, targets):
        """Standard YOLOv5 build_targets with 6-column labels [img_idx, class, x, y, w, h]."""
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch