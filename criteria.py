import torch
import torch.nn as nn
import torch.nn.functional as F

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    

class HuberLoss(nn.Module):
    def __init__(self, delta=19):  # 290开根号
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        error = target - pred
        valid_mask = (target > 0).detach()
        error = error[valid_mask]
        # th_mask = (torch.abs(error) > 1).detach()
        # error = error(th_mask)
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * torch.abs(error)**2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        self.loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        return self.loss.mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = torch.log(torch.cosh(diff))
        return self.loss.mean()
    
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        # Ensure the inputs and targets have the same shape
        if inputs.shape != targets.shape:
            raise ValueError("Shape of inputs and targets must match")

        # Create a mask where targets > 0
        mask = targets > 0

        # Sigmoid function to get probabilities
        inputs_sigmoid = torch.sigmoid(inputs)

        # Compute binary cross-entropy loss manually
        BCE_loss = - (targets * torch.log(inputs_sigmoid + 1e-8) + (1 - targets) * torch.log(1 - inputs_sigmoid + 1e-8))

        # Apply the mask to the loss
        BCE_loss = BCE_loss * mask

        # Compute the focal loss scaling factor
        pt = torch.exp(-BCE_loss)  # p_t = exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # Apply the mask to the focal loss
        F_loss = F_loss * mask

        if self.reduction == 'mean':
            return F_loss.sum() / mask.sum()  # Normalize by the number of valid elements
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss   

class Huber_Combine(nn.Module):
    def __init__(self, delta=17):  # 290开根号
        super(Huber_Combine, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        L1 = 0.7
        L2 = 0.3
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        error = target - pred
        valid_mask = (torch.abs(target) > 0).detach()
        error = error[valid_mask]
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        loss_1 = torch.where(is_small_error, small_error_loss, large_error_loss).mean()
        loss_2 = (error**2).mean()
        self.loss = L1 * loss_1 + L2 * loss_2
        return self.loss
    

class info_nce_loss_depth_maps(nn.Module):
    """
    计算两张深度图的 InfoNCE 损失。
   
    参数:
    - depth_map1: Tensor，第一张深度图的张量表示，形状为 (H, W)。
    - depth_map2: Tensor，第二张深度图的张量表示，形状为 (H, W)。
    - temperature: float，温度参数，控制相似度分布的平滑程度。
   
    返回:
    - loss: Tensor，InfoNCE 损失值。
    """
    def __init__(self):
        super(info_nce_loss_depth_maps, self).__init__()
        self.temperature = 0.5
        
    def forward(self, depth1_pred, depth2_pred):
        # 将深度图展平为一维向量
        depth1_pred= depth1_pred.flatten() # (H * W,)
        depth2_pred= depth2_pred.flatten() # (H * W,)

        # 将展平后的深度图视为特征向量，并进行归一化
        depth_map1 = F.normalize(depth1_pred.unsqueeze(0), dim=1)  # (1, H * W)
        depth_map2 = F.normalize(depth2_pred.unsqueeze(0), dim=1)  # (1, H * W)

        # 计算正样本对的相似性
        positive_similarity = torch.sum(depth_map1 * depth_map2) / self.temperature  # 标量

        # 计算负样本对的相似性矩阵
        all_features = torch.cat([depth_map1, depth_map2], dim=0)  # (2, H * W)
        similarity_matrix = torch.mm(all_features, all_features.T) / self.temperature  # (2, 2)

        # 掩码排除自身对自身的相似性
        mask = torch.eye(2, dtype=torch.bool).to(depth_map1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # 构建 logits 并计算损失
        logits = torch.cat([positive_similarity.unsqueeze(0), similarity_matrix[0, 1].unsqueeze(0)], dim=0)
        labels = torch.zeros(1, dtype=torch.long).to(depth_map1.device)

        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss



