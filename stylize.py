import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from PIL import Image
import copy
import argparse

from loss import StyleLoss, ContentLoss
from utils import save_image, Normalization, image_loader, toPIL
from vgg import load_vgg16
import sys
import getopt

from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Working on ", device)

imsize = 800 if torch.cuda.is_available() else 128

normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
normalization = Normalization(normalization_mean, normalization_std)
normalization = normalization.to(device)

content_layers_default = ['conv1', 'conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(parent_model,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    parent_model = copy.deepcopy(parent_model)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in parent_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'
                               .format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or \
           isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(parent_model,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):

    out = get_style_model_and_losses(parent_model, style_img, content_img)
    model, style_losses, content_losses = out
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    print('Starting..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("Epoch {}/{}:".format(run[0], num_steps), end='\t')
                print('Style Loss : {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


def stylize(content_img, style_img, image_size=imsize, weight=10000, epochs=300):
    modelfile = './.saved_models/vgg16-397923af.pth'
    cnn = load_vgg16(modelfile)
    cnn = cnn.features.to(device).eval()

    input_img = content_img.clone()

    output = run_style_transfer(cnn, content_img, style_img, input_img, style_weight=weight, num_steps=epochs)

    output = output.cpu()
    output = output.squeeze(0)
    output = toPIL(output)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stylize Image using VGG16, Gatys et al")
    parser.add_argument('-i', '-image', type=str, required=True, help="Image Content")
    parser.add_argument('-s', '-style', type=str, required=True, help="Image Style")
    parser.add_argument('--output', type=str, default="output.jpg", help="Location of output file")
    parser.add_argument('--w', type=int, default=100000, help="Style Weight in Loss Computation")
    parser.add_argument('--epochs', type=int, default=300, help="Style Weight in Loss Computation")

    args = parser.parse_args()

    print('Input file is ', args.i)
    print('Style file is ', args.s)

    style_img = image_loader(args.s, imsize, device)
    content_img = image_loader(args.i, imsize, device)

    output = stylize(content_img, style_img, weight=args.w, epochs=args.epochs)
    output.save(args.output)
