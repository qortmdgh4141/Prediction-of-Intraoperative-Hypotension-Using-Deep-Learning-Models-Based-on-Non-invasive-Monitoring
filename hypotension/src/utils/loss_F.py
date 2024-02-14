import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops.focal_loss import sigmoid_focal_loss


def call_loss_fn(loss_name, **kwargs):
    if loss_name.lower() == 'tilt':
        if kwargs['Tau'] == 0.0:
            criterion = Tilted_Loss_(t=kwargs['T'])
        else:
            criterion = Tilted_Loss(t=kwargs['T'], tau=kwargs['Tau'])
    elif loss_name.lower() == 'focal':
        if 'gamma' not in kwargs.keys():
            raise ValueError('Gamma value is missing for focal loss')
        elif 'alpha' not in kwargs.keys():
            raise ValueError('Alpha value is missing for focal loss')
        else:
            criterion = FocalLoss(gamma=kwargs['gamma'], alpha=kwargs['alpha'], size_average=False)
    elif loss_name.lower() == 'cfocal':
        criterion = CfocalLoss(gamma=2)
    elif loss_name.lower() == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()
    return criterion


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

## gamma: 2.0 alpha: [0.5, 0.5]
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input.to(dtype=torch.float64)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CfocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, **kwargs):
        super(CfocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):

        # import keras.backend as K
        # epsilon = K.epsilon()
        # y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        #
        #

        batch_size = y_pred.shape[0]

        y_pred = torch.exp(y_pred) / torch.sum(torch.exp(y_pred), dim=1, keepdim=True)  # do softmax
        y_pred = torch.clamp(y_pred, 1e-13, 1 - 1e-13)  # clipping

        weight = torch.pow((1 - y_pred), self.gamma)
        weighted_cross_entropy = -torch.log(y_pred) * weight

        result = weighted_cross_entropy[range(batch_size), y_true]  # select value whose label y_true is true
        loss = torch.sum(result) / batch_size  # sum and average

        return loss


##new tilted loss
class Tilted_Loss(nn.Module):
    def __init__(self, tau=-0.3, t=4.3, n_classes=2):
        super(Tilted_Loss, self).__init__()
        self.n_classes = n_classes
        self.tau = tau
        self.t = t

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        inds = [[i for i in range(batch_size) if k == y_true[i]] for k in range(self.n_classes)]  ##index of each class.

        soft_yp = torch.exp(y_pred) / torch.sum(torch.exp(y_pred), dim=1, keepdim=True)  # do softmax
        soft_yp = torch.clamp(soft_yp, 1e-13, 1 - 1e-13)  # clipping

        cross_entropy = -torch.log(soft_yp)

        reverse_cross_entropy_tau = torch.exp(
            cross_entropy * self.tau)  ####reverse cross entropy with tau tilting, remove noise

        loss = torch.zeros(1).cuda()  ##total class loss

        ##t-tilt on each class and averaging
        for k in range(self.n_classes):
            if len(inds[k]) != 0:
                a = reverse_cross_entropy_tau[inds[k], [k for i in range(len(inds[k]))]] / len(inds[k])
                average_k = torch.sum(a)  ###average soft_yp value on class k with tau-tilting

                loss_k = 1 / self.tau * torch.log(average_k)  # tau_tilted_loss of class k

                loss += len(inds[k]) * torch.exp(
                    loss_k * self.t) / batch_size / 1000000000000  ##inter class loss averaging with t-tilting

        return loss


##use??
class Tilted_Loss_(nn.Module):
    def __init__(self, t=4.3, n_classes=2):
        super(Tilted_Loss_, self).__init__()
        self.n_classes = n_classes
        self.t = t

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]


        inds = [[i for i in range(batch_size) if k == y_true[i]] for k in range(self.n_classes)]  ##index of each class.

        batch_size = torch.Tensor([batch_size]).type(torch.float64).to('cuda')

        soft_yp = torch.exp(y_pred) / torch.sum(torch.exp(y_pred), dim=1, keepdim=True)  # do softmax
        soft_yp = torch.clamp(soft_yp, 1e-13, 1 - 1e-13)  # clipping

        cross_entropy = -torch.log(soft_yp).type(torch.float64)

        loss = torch.zeros(1).cuda()  ##total class loss
        loss = loss.to(dtype=torch.float64)

        ##t-tilt on each class and averaging
        for k in range(self.n_classes):
            if len(inds[k]) != 0:
                a = cross_entropy[inds[k], [k for i in range(len(inds[k]))]] / torch.Tensor([len(inds[k])]).type(torch.float64).to('cuda')
                average_k = torch.sum(a)  ###average soft_yp value on class k with tau-tilting

                loss_k = average_k  # tau_tilted_loss of class k

                loss += len(inds[k]) * torch.exp(
                    loss_k * self.t) / batch_size / torch.Tensor([1000000000000]).type(torch.float64).to('cuda')  ##inter class loss averaging with t-tilting

        return loss
