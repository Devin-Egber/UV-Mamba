import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


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
            target_flat = ground_truth.view(-1, 1)
            logits_flat = logits.permute(0, 2, 3, 1).flatten(0, 2)
            # 这里是要忽略计算loss的类别
            mask_flat = target_flat[:, 0] != 255
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



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply softmax to the predictions
        y_pred = F.softmax(y_pred, dim=1)

        # Flatten the tensors
        y_pred = y_pred[:, 1, ...].contiguous().view(-1)  # Use the channel corresponding to class 1
        y_true = y_true.contiguous().view(-1).float()  # Flatten the ground truth

        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Compute Dice loss
        loss = 1 - dice

        return loss


def get_loss(dataset_config):
    if dataset_config.dataset == 'uvseg':

        if dataset_config.city == 'beijing':
            criterion = DiceLoss()
        if dataset_config.city == 'xian':
            criterion = nn.CrossEntropyLoss()

    elif dataset_config.dataset == 'cityscapes':
        criterion = MaskedCrossEntropyLoss()
    else:
        raise NotImplementedError

    return criterion

