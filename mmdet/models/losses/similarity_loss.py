import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def cosin_loss(pred, target,delta=0.1):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    #loss = torch.abs(pred - target)
    numer = torch.sum(pred * target, dim=1).view((pred.size()[0],-1))
    denom = torch.sqrt(torch.sum(pred ** 2, dim=1) * torch.sum(target ** 2, dim=1)).view((pred.size()[0],-1))
    similar = numer / (denom+delta)  # 这实际是夹角的余弦值
    #loss = 1-(similar + 1) / 2  # 姑且把余弦函数当线性
    loss = 1 - similar
    return loss



@LOSSES.register_module()
class CosinLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CosinLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * cosin_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * cosin_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox






@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def cosin_dis(pred, target,delta=0.1):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    #loss = torch.abs(pred - target)
    numer = torch.sum(pred * target, dim=1).view((pred.size()[0],-1))
    denom = torch.sqrt(torch.sum(pred ** 2, dim=1) * torch.sum(target ** 2, dim=1)).view((pred.size()[0],-1))
    similar = numer / (denom+delta)  # 这实际是夹角的余弦值
    #loss = 1-(similar + 1) / 2  # 姑且把余弦函数当线性
    loss = similar #1 - similar
    return loss



@LOSSES.register_module()
class CosinDis(nn.Module):
    """CosinDis.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CosinDis, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * cosin_dis(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * cosin_dis(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox








@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def similar2_loss(pred, target,delta=0.1):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    differ = target -pred
    numera = torch.sum(differ ** 2, dim=1).view((pred.size()[0],-1))
    denom = torch.sum(target ** 2, dim=1).view((pred.size()[0],-1))
    similar = 1 - (numera / (denom+delta))

    loss = 1 - similar
    return loss



@LOSSES.register_module()
class Similar2Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(Similar2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * similar2_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * similar2_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def similar3_loss(pred, target,delta=0.1):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    #loss = torch.abs(pred - target)
    differ = pred - target
    dist = torch.norm(differ, p='fro', dim=1).view((pred.size()[0],-1))
    len1 = torch.norm(pred, p='fro', dim=1).view((pred.size()[0],-1))
    len2 = torch.norm(target, p='fro', dim=1).view((pred.size()[0],-1))  # 普通模长
    denom = (len1 + len2) / 2
    similar = 1 - (dist / (denom+delta))

    loss = 1 - similar
    return loss





@LOSSES.register_module()
class Similar3Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(Similar3Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * similar3_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * similar3_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ssim_loss(pred, target,delta=0.1):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    np= pred.size()[1]
    mean_pred = torch.sum(pred, dim=1).view((pred.size()[0],-1))/np
    mean_target = torch.sum(target, dim=1).view((target.size()[0], -1)) / np
    var_pred = torch.diag(torch.mm(pred-mean_pred, (pred-mean_pred).t())).view((pred.size()[0],-1)) /(np-1)
    var_target = torch.diag(torch.mm(target-mean_target, (target-mean_target).t())).view((pred.size()[0],-1)) /(np-1)
    std_pt = torch.diag(torch.mm(pred-mean_pred,(target-mean_target).t())).view((pred.size()[0],-1)) /(np-1)

    a1 = 2 * mean_pred*mean_target +delta
    a2 = 2 * std_pt + delta
    b1 = mean_pred ** 2 + mean_target ** 2 + delta
    b2 = var_pred + var_target + delta
    similar = (a1*a2) / (b1*b2+delta)  # 这实际是夹角的余弦值
    #loss = 1-(similar + 1) / 2  # 姑且把余弦函数当线性
    loss = 1 - similar
    return loss



@LOSSES.register_module()
class SsimLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SsimLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * ssim_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * ssim_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox






@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ssim_dis(pred, target,delta=0.1):
    """ssim_dis.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    np= pred.size()[1]
    mean_pred = torch.sum(pred, dim=1).view((pred.size()[0],-1))/np
    mean_target = torch.sum(target, dim=1).view((target.size()[0], -1)) / np
    var_pred = torch.diag(torch.mm(pred-mean_pred, (pred-mean_pred).t())).view((pred.size()[0],-1)) /(np-1)
    var_target = torch.diag(torch.mm(target-mean_target, (target-mean_target).t())).view((pred.size()[0],-1)) /(np-1)
    std_pt = torch.diag(torch.mm(pred-mean_pred,(target-mean_target).t())).view((pred.size()[0],-1)) /(np-1)

    a1 = 2 * mean_pred*mean_target +delta
    a2 = 2 * std_pt + delta
    b1 = mean_pred ** 2 + mean_target ** 2 + delta
    b2 = var_pred + var_target + delta
    similar = (a1*a2) / (b1*b2+delta)  # 这实际是夹角的余弦值
    #loss = 1-(similar + 1) / 2  # 姑且把余弦函数当线性
    loss = similar #1 - similar
    return loss



@LOSSES.register_module()
class SsimDisLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SsimDisLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * ssim_dis(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * ssim_dis(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox








@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def matrelat_loss(pred, target,delta=0.3):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0

    differ = target - pred
    #loss_l2 = torch.sqrt(torch.sum(differ ** 2, dim=1).view((pred.size()[0], -1))) #矩阵F范数
    loss_l2 = torch.norm(differ, p='fro', dim=1).view((pred.size()[0], -1))  #矩阵F范数
    loss_l2 = loss_l2 *loss_l2

    relat = (pred.unsqueeze(1) - pred.unsqueeze(0)) - (target.unsqueeze(1) - target.unsqueeze(0))
    loss_rel = torch.sqrt(torch.sum(relat ** 2, dim=2).view((pred.size()[0], -1)))
    loss_rel = torch.sum((loss_rel * loss_rel ), dim=1).view((pred.size()[0], -1))


    loss = loss_l2 + loss_rel*delta
    return loss



@LOSSES.register_module()
class MatRelatLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MatRelatLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if torch.any(torch.isnan(self.loss_weight * matrelat_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor))) \
                or torch.any(torch.isnan(pred)) or torch.any(torch.isnan(target)):
            loss_bbox = torch.ones_like(target).sum() / (target.size()[0] * target.size()[1])
        else:
            loss_bbox = self.loss_weight * matrelat_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox