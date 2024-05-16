import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "imputation":
        return 

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def corrcoeff(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask:torch.BoolTensor,) -> torch.Tensor:
        """
        Pearson correlation coefficient loss.
        """
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)

        y_pred_mean = torch.mean(y_pred, dim=0, keepdim=True)
        y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
        centered_y_pred = y_pred - y_pred_mean
        centered_y_true = y_true - y_true_mean

        covariance = (1 / (y_pred.shape[0] - 1)) * torch.matmul(centered_y_pred.t(), centered_y_true)

        y_pred_std = torch.std(y_pred, dim=0, keepdim=True)
        y_true_std = torch.std(y_true, dim=0, keepdim=True)

        y_pred_std = torch.clamp(y_pred_std, min=1e-8)
        y_true_std = torch.clamp(y_true_std, min=1e-8)

        correlation = covariance / (y_pred_std * y_true_std)

        return torch.abs(correlation)


    def nrmse(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

            
        ADDITION: I now add the correlation and the MSE loss to a combined metric.


        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        nrmse = torch.divide(self.mse_loss(masked_pred, masked_true).sqrt(), self.mse_loss(masked_pred, masked_true).sqrt().max())
        return nrmse


    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

            
        ADDITION: I now add the correlation and the MSE loss to a combined metric.


        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        nrmse = self.nrmse(y_pred=y_pred, y_true=y_true, mask=mask)
        corrcoeff = self.corrcoeff(y_pred=y_pred, y_true=y_true, mask=mask)

        return nrmse # * (1 - corrcoeff)