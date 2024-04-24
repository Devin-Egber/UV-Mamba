import torch.nn.functional as F
from torch import nn
import torch


class FocalCELoss(nn.Module):
    def __init__(self, gamma=1.0, size_average=True, ignore_index = -100, weight = None):
         super(FocalCELoss, self).__init__()
         self.gamma = gamma
         self.size_average = size_average
         self.ignore_index = ignore_index
         self.weight = weight

    def forward(self, preds, target):
        target = target.view(-1,1)
        if preds.ndim > 2: # e.g., (B, C, H, W) ---> (B, H, W, C) ---> (B * H * W, C)
            preds = preds.permute(0, 2, 3, 1).flatten(0, 2)
        keep = target[:, 0] != self.ignore_index
        preds = preds[keep, :]
        target = target[keep, :]
        logpt = F.log_softmax(preds, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.weight is not None:
            w = self.weight.expand_as(preds)
            w = w.gather(1, target)
            loss = -1 * (1 - pt) ** self.gamma * w * logpt
        else:
            loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, mean=True, mask=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean
        self.mask = mask

    def forward(self, logits, ground_truth):
        if self.mask:
            target_flat = ground_truth.view(-1,1)
            logits_flat = logits.permute(0, 2, 3, 1).flatten(0, 2)
            mask_flat = target_flat[:, 0] != 19
            masked_logits_flat = logits_flat[mask_flat, :]
            masked_target_flat = target_flat[mask_flat, :]
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_target_flat = ground_truth.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)

        masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss



def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss