import torch
from torch import nn
from torchvision.models.vgg import vgg16


class L2Loss(nn.Module):
    def __init__(self, loss_weight=1):
        """
        L2 loss for regularization loss
        """
        super(L2Loss, self).__init__()
        self.loss_weight = loss_weight

    def calc_size(self, t: torch.Tensor):
        """
        size = channels * height * width
        """
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        x_h = x.size()[2]
        x_w = x.size()[3]
        h_c = self.calc_size(x[:, :, 1:, :])
        w_c = self.calc_size(x[:, :, :, 1:])
        h_l2 = torch.pow((x[:, :, 1:, :] - x[:, :, :x_h - 1, :]), 2).sum()
        w_l2 = torch.pow((x[:, :, :, 1:] - x[:, :, :, :x_w - 1]), 2).sum()

        return self.loss_weight * 2 * (h_l2 / h_c + w_l2 / w_c) / batch_size


class GeneratorLoss(nn.Module):
    def __init__(self):
        """
        perceptual loss =
            l_pix + (1e-3 * l_adv) + (6e-3 * l_vgg) + (2e-8 * l_reg)
        """
        super(GeneratorLoss, self).__init__()
        model_vgg = vgg16(pretrained=True)
        loss_vgg = nn.Sequential(*list(model_vgg.features)[:31]).eval()
        for param in loss_vgg.parameters():
            param.requires_grad = False

        self.loss_vgg = loss_vgg
        self.loss_pix = nn.MSELoss()
        self.l2loss = L2Loss()

    def forward(self, x):
        pass
