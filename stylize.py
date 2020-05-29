import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from PIL import Image
import copy

from loss import StyleLoss, ContentLoss
from utils import save_image, Normalization, image_loader
from vgg import load_vgg16
import sys
import getopt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Working on ", device)

imsize = 800 if torch.cuda.is_available() else 128
print("imsize set to", imsize)
print()

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
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    out = get_style_model_and_losses(parent_model, style_img, content_img)
    model, style_losses, content_losses = out
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    save_epoch = False

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
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                if save_epoch:
                    outputfile = './art-{}.jpg'.format(run[0])
                    save_image(outputfile, input_img)
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == "__main__":
    argv = sys.argv[1:]
    modelfile = './.saved_models/vgg16-397923af.pth'
    outputfile = './art.jpg'
    inputfile = "./images/dancing.jpg"
    stylefile = "./images/picasso.jpg"
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "sfile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <stylefile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python stylize.py -i <inputfile> -s <stylefile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-s", "--sfile"):
            stylefile = arg
    print('Input file is "', inputfile)
    print('Style file is "', stylefile)
    cnn = load_vgg16(modelfile)
    cnn = cnn.features.to(device).eval()

    style_img = image_loader(stylefile, imsize, device)
    content_img = image_loader(inputfile, imsize, device)
    input_img = content_img.clone()
    USE_WHITE_NOISE = False
    if USE_WHITE_NOISE:
        input_img = torch.randn(content_img.data.size(), device=device)

    output = run_style_transfer(cnn, content_img, style_img, input_img)

    save_image(outputfile, output)
