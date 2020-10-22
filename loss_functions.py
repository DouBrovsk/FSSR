import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def perceptionLoss(pretrained_model="vgg16", device=device):
    """
    Possible parameters: "vgg16", "resnet18", "vgg19"
    """
    if pretrained_model == "vgg16":
        m = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features)[:9]).to(device)
    elif pretrained_model == "resnet18":
        tmp_res = torchvision.models.resnet18(pretrained=True)
        m = nn.Sequential(tmp_res.conv1, tmp_res.bn1, tmp_res.relu, tmp_res.maxpool, tmp_res.layer1).to(device)
        del tmp_res
    elif pretrained_model == "vgg19":
        m = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features)[:9]).to(device)
    else:
        raise Exception("Error: Unknown model for perception loss function.")

    def loss(x, y): # Differentiable and the weights of the loss network won't be updated because they were not passed as an argument to the optimizer
        return torch.mean((m(x)-m(y))**2)
    return loss

def reconstructionLoss():
    pass # ToDo


def ultimateLoss(pretrained_model="vgg16", device=device, alpha=0.95, beta=0.05):
    percLoss = perceptionLoss(pretrained_model=pretrained_model, device=device)
    mse = nn.MSELoss()
    def loss(x, y):
        return alpha*mse(x, y) + beta*percLoss(x, y)

    return loss











