import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image, ImageEnhance
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ToTensor:
    def __call__(self, img, mask):
        return F.to_tensor(img), F.to_tensor(mask).squeeze_(0).type(torch.int64)


class Normalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, mask):
        return F.normalize(img, self.mean, self.std), mask


class RandomCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img, mask):
        assert img.size == mask.size

        if img.size == self.size:
            return img, mask

        W, H = self.size
        w, h = img.size
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return img.crop(crop), mask.crop(crop)


class RandomScale:
    def __init__(self, scales=(0.75, 1.25)):
        self.scales = scales if isinstance(scales, tuple) else list(scales)

    def __call__(self, image, mask):
        W, H = image.size
        if isinstance(self.scales, tuple):
            scale = random.random() * (self.scales[1] - self.scales[0]) + self.scales[0]
        else:
            scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return image.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)


class RandomRotation:
    def __init__(self, degrees=(-30, 30)):
        self.degrees = degrees

    def __call__(self, img: Image.Image, mask: Image.Image):
        degree = random.randint(self.degrees[0], self.degrees[1])
        return img.rotate(degree), mask.rotate(degree, resample=Image.NEAREST)


class ColorJitter:
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if brightness is not None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if contrast is not None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if saturation is not None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, image, mask):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        image = ImageEnhance.Brightness(image).enhance(r_brightness)
        image = ImageEnhance.Contrast(image).enhance(r_contrast)
        image = ImageEnhance.Color(image).enhance(r_saturation)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img, mask = F.hflip(img), F.hflip(mask)
            mask_np = np.asarray(mask).copy()
            for a, b in [(2, 3), (4, 5), (7, 8)]:
                ma, mb = mask_np == a, mask_np == b
                mask_np[ma], mask_np[mb] = b, a
            mask = Image.fromarray(mask_np)
        return img, mask
