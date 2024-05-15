"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        # g_neg = 2  g_pos = 0   clip = 0  eps = 1e-5    true
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, weight_list = None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        #temp(temp < 0) = 0
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0: # not access
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)


                    weight = torch.ones_like(self.asymmetric_w)
                    if weight_list is not None:
                        # dynamic loss
                        weight_list = torch.from_numpy(weight_list).cuda()
                        weight = weight * weight_list
                        # free mem
                        weight_list.cpu()
                        torch.cuda.empty_cache()
                        for index in range(len(weight_list)):
                            for index_1 in range(weight.shape[0]):
                                weight[index_1][index] = weight_list[index]
                    else:
                    # static loss

                        bottle_class = [0, 5, 10, 14]
                        # weight = torch.ones_like(self.asymmetric_w)
                        # [bs, 17]
                        for index in bottle_class:
                            for index_1 in range(weight.shape[0]):
                                weight[index_1][index] *= 2

                    self.asymmetric_w *= weight

                self.loss *= self.asymmetric_w

            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w
        # self.loss : [16,17]
        # sum mul bs: become a number
        _loss = - self.loss.sum() / x.size(0)

        _loss = _loss / y.size(1) * 1000

        return _loss


class Ralloss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0,
                 epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False):
        super(Ralloss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.margin = 1.0
        self.lamb = lamb

    def forward(self, x, y, weight_list=None):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (
                    1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(
            1 - xs_pos.clamp(min=self.eps), 2))
        los_neg = (1 - y) * (
                    torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps))) * (
                              self.lamb - x_sigmoid) * x_sigmoid ** 2 * (self.lamb - xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            weight = torch.ones_like(one_sided_w)
            bottle_class = [0, 5, 10, 14]
            for index in bottle_class:
                for index_1 in range(weight.shape[0]):
                    weight[index_1][index] =weight[index_1][index] * 2

            one_sided_w =one_sided_w * weight

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
