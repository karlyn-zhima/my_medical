import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ProgLoss','BCEDiceLoss','DiceLoss']

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class BCEDiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes=n_classes

    def forward(self, input, target):
        input=torch.argmax(input,dim=1).float()
        target=target.float()
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class ProgLoss(nn.Module):
    def __init__(self,epoch=800):
        super().__init__()
        self.epoch=epoch

    def forward(self, out, label, cl, cls,epoc):
        index1=[torch.argmax(cls,dim=1)==0]
        index2=[torch.argmax(cls,dim=1)==1]
        
        weight = torch.FloatTensor([1,1]).cuda()

        loss_div=torch.nn.L1Loss()(cl,cls)
        loss_conv=self.convexityloss(out)
        a=(epoc+1)/(self.epoch-500)
        b=max(0.1,1-a)
        loss = torch.nn.CrossEntropyLoss(weight=weight)(out[index1],label[index1])+0.2*pow(5,b)*torch.nn.CrossEntropyLoss(weight=weight)(out[index2],label[index2])

        return b*loss_div+a*loss+a*loss_conv,loss_div,loss_conv
