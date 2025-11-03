import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Dice, Precision, Recall, F1Score, Accuracy
from typing import Any, Optional, Tuple
from torch import distributed
from typing import Callable
import cv2
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val==-1:
            return 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice(preds,target):
    """2TP / (2TP + FP + FN)"""
    dice_m = Dice(average='micro',ignore_index=0).cuda() # exclude the background
    return dice_m(preds,target)

def precision(preds,target,task='binary'):
    """TP / (TP + FP)"""
    precision_m = Precision(task=task).cuda()
    return precision_m(preds,target)

def recall(preds,target,task='binary'):
    """TP / (TP + FN)"""
    recall_m = Recall(task=task).cuda()
    return recall_m(preds,target)

def fscore(preds,target,task='binary'):
    f1_m = F1Score(task=task).cuda()
    return f1_m(preds,target)

def accuracy(preds,target,task='binary'):
    acc_m = Accuracy(task=task).cuda()
    return acc_m(preds,target)

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Symmetric Cross-Entropy loss for binary segmentation with shape (B, N),
        where B is the batch size and N is the number of pixels.
        
        Parameters:
        - alpha: Weight for the CE loss.
        - beta: Weight for the RCE loss.
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha  # Weight for CE loss
        self.beta = beta    # Weight for RCE loss

    def forward(self, logits, labels):
        """
        Args:
            logits: Tensor of shape (batch_size, N) - raw model outputs (before sigmoid)
            labels: Tensor of shape (batch_size, N) - ground truth labels (0 or 1)

        Returns:
            loss: A scalar value representing the Symmetric Cross-Entropy loss
        """
        # Apply sigmoid to logits to get probabilities (for each pixel)
        probs = torch.sigmoid(logits)  # shape: (batch_size, N)

        # Convert labels to float for loss computation (0 or 1)
        labels = labels.float()  # shape: (batch_size, N)

        # Cross-Entropy (CE) Loss
        ce_loss = -torch.mean(
            labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8)
        )

        # Reverse Cross-Entropy Loss (RCE)
        rce_loss = -torch.mean(
            probs * torch.log(labels + 1e-8) + (1 - probs) * torch.log(1 - labels + 1e-8)
        )

        # Combine both losses
        return self.alpha * ce_loss + self.beta * rce_loss

class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None

class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        mask = calc_boundary_att(target, t, T=1000, gamma=0.2) # (B,1,256,256)

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

def calc_edge(x, mode='canny'):
    x = np.uint8(x)
    edge = cv2.Canny(image=x, threshold1=0, threshold2=0)
    return edge

def calc_distance_map(x, mode='l2'):
    # if isinstance(x, torch.Tensor):
    #     x = x.numpy()
    
    # Convert the data to grayscale [0,255]
    binary_x = 1 - np.uint8((x-x.min())/(x.max()-x.min()))

    if mode.lower() == 'l1':
        dt_mode = cv2.DIST_L1
    elif mode.lower() == 'l2':
        dt_mode = cv2.DIST_L2
    else:
        raise ValueError("<mode> must be 'l1' or 'l2'.")
    
    # Calculate the distance transform
    dist_transform= cv2.distanceTransform(binary_x, dt_mode, 0)

    return dist_transform

def calc_boundary_att(batch_x, batch_t, T, gamma=1.5, *args, **kwargs):
    """
    Parameters:
        - batch_x : [tensor] |-> input data matrix
        - batch_t : [tensor] |-> current timestep
        - T       : [int]    |-> maximum timesteps
        - gamma   : [float]  |-> sharpness [default is 1.5]
    Output:
        - boundary_att
    """
    
    
    # boundary thickness (max value thickness)
    bt = np.round(batch_x.shape[-1]*0.01)
    
    X = batch_x.detach().cpu().numpy()
    X = (X-X.min())/(X.max()-X.min()) # normalize because X is in range ~ (-1, 1)
    device = batch_x.get_device()
    
    atts = []
    for x, t in zip(X, batch_t):
        if x.sum().item() > X.shape[-1]**2/100.: # foreground area is bigger than 1% of the image
            x = (x-x.min())/(x.max()-x.min())
            edge = calc_edge(x[0,:])
            dist_x = calc_distance_map(edge, mode='l2')
            tmp = X.shape[-1]*1.1415 - dist_x
            normalized_inv_dist_x = (tmp-tmp.min())/ (tmp.max()-tmp.min())
            
            t_p = ((gamma*(T-t.item()))/T)**gamma
            att = normalized_inv_dist_x**t_p            
        else:
            att = np.ones_like(x[0,:])
        atts.append(att)

    atts = np.array(atts)
    
    W = torch.stack([torch.from_numpy(att) for att in atts], dim=0)
    if device != -1:
        W = W.to(device)

    W = torch.unsqueeze(W, dim=1)
    return W

class BoundaryLoss(torch.nn.Module):
    """Some Information about BoundaryLoss"""
    def __init__(self, parameters={}):
        super().__init__()
        # self.logger = get_logger()
        
        self.gamma=parameters.get("gamma", 1.5)
        root=parameters.get("root", "l2")
        if root == "l2":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)**2
        elif root == "l1":
            self.calc_root_loss = lambda p, t: torch.abs(p-t)
        # else:
        #     self.logger.exception("Not implemented!")

    def forward(self, x, t, T, predicted_noise, noise):
        boundary_att = calc_boundary_att(x, t, T=T, gamma=self.gamma) # (B,1,256,256)
        root_loss = self.calc_root_loss(predicted_noise, noise)
        return (boundary_att * root_loss).mean(), boundary_att