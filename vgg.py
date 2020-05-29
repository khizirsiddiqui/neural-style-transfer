import torch
from torchvision import models


def load_vgg16(PATH):
    net = models.vgg16(pretrained=False)
    net.load_state_dict(torch.load(PATH))
    return net


if __name__ == "__main__":
    PATH = './.saved_models/vgg16-397923af.pth'
    vgg16 = load_vgg16(PATH)
    print("Sucessfully Loaded")
    print(vgg16.features)
