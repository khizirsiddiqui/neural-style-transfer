import torch
from torch import nn
from PIL import Image
from torchvision import transforms

toPIL = transforms.ToPILImage()

def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def load_image(filename, size=None, scale=None):
    """Load image from a given address

    Arguments:
        filename {str} -- Filename of image.

    Keyword Arguments:
        size {int} -- Resize image to this size. (default: {None})
        scale {float} -- Resize and then rescale image. (default: {None})

    Returns:
        Image after resizing and rescaling (if any).
    """
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    if scale is not None and size is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    return img


def save_image(filename, tensor):
    """Save image to the given address

    Arguments:
        filename {[str]} -- Filename
        data {[matrix]} -- Image data
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = toPIL(image)
    image.save(filename)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
