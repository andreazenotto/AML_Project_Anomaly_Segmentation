import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
    

class IsoMaxPlusLoss(nn.Module):
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLoss, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, outputs, targets):
        distances = -outputs
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = torch.gather(probabilities_for_training, 1, targets.unsqueeze(1))
        probabilities_at_targets = torch.clamp(probabilities_at_targets, min=1e-7)
        loss = -torch.log(probabilities_at_targets).mean()
        return loss
        

class LogitNormLoss(nn.Module):

    def __init__(self, t=0.01):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, outputs, target):
        norms = torch.norm(outputs, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(outputs, norms) / self.t
        return F.cross_entropy(logit_norm, target)
    

class FocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        targets = F.one_hot(targets, num_classes=outputs.size(1)).float()
        targets = targets.permute(0,3,1,2)
        p_t = (outputs * targets).sum(dim=1)
        loss = -self.alpha * (1 - p_t)**self.gamma * torch.log(p_t)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class CombinedLoss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0, w3=1.0, w4=1.0):
        super(CombinedLoss, self).__init__()
        self.iso_max = IsoMaxPlusLoss()
        self.logit_norm = LogitNormLoss()
        self.focal_loss = FocalLoss()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        
        # Weights for each loss
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def forward(self, outputs, targets):
        loss_iso = self.iso_max(outputs, targets)
        loss_logit = self.logit_norm(outputs, targets)
        loss_focal = self.focal_loss(outputs, targets)
        loss_ce = self.cross_entropy_loss(outputs, targets)
        
        # Weighted sum of the losses
        total_loss = (
            self.w1 * loss_iso +
            self.w2 * loss_logit +
            self.w3 * loss_focal +
            self.w4 * loss_ce
        )
        return total_loss
