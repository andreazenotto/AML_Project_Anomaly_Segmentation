import torch
import torch.nn as nn
import torch.nn.functional as F

class IsoMaxPlusLoss(nn.Module):
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLoss, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, outputs, targets):
        distances = -outputs
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        return loss
        
# References:
# https://arxiv.org/pdf/2105.14399
# https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/master/losses/isomaxplus.py

# --------------------------------------------------------------------------------------------------- #

class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, outputs, target):
        norms = torch.norm(outputs, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(outputs, norms) / self.t
        return F.cross_entropy(logit_norm, target)
    
# References:
# https://arxiv.org/pdf/2205.09310
# https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py

# --------------------------------------------------------------------------------------------------- #

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets = F.one_hot(targets, num_classes=outputs.size(1)).float()
        p_t = (outputs * targets).sum(dim=1)
        loss = -self.alpha * (1 - p_t)**self.gamma * torch.log(p_t)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
# References:
# https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
