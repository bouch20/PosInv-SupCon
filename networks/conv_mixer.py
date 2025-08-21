import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 100, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=100)
    model.default_cfg = _cfg
    return model[:-1]


@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 8, kernel_size=7, patch_size=7, n_classes=100)
    model.default_cfg = _cfg
    return model[:-1]

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    # nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=0),
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=[kernel_size//2, kernel_size // 2], padding_mode='replicate'),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

