"""Commonly used normalize method in Generative Adversarial Networks. """
import torch


def pixel_norm(x, eps=1e-8):
    """Pixelwise feature vector normalization. """
    return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)


def instance_norm(x, eps=1e-8):
    """Instance normalization. """
    assert len(x.shape) == 4, "shape of input should be NCHW!"
    x -= torch.mean(x, dim=(2, 3), keepdim=True)
    return x * torch.rsqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + eps)


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    """Minibatch standard devation. """
    group_size = min(group_size, x.size(0)) # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                             # [NCHW]  Input shape.
    y = torch.reshape(x, (group_size, -1, num_new_features, s[1] // num_new_features, *s[2:])) # [GMncHW]
    y -= torch.mean(y, dim=0, keepdim=True) # [GMncHW] Subtract mean over group.
    y = torch.mean(y ** 2, dim=0)           # [MncHW]  Calc variance over group.
    y = torch.sqrt(y + 1e-8)                # [MncHW]  Calc stddev over group.
    y = torch.mean(y, dim=(2, 3, 4), keepdim=True) # [Mn111] Take average over fmaps and pixels.
    y = torch.mean(y, dim=2)                       # [Mn11]  Split channels into c channel groups.
    y = y.repeat(group_size, num_new_features, *s[2:]) # [NnHW] Replicate over group and pixels.

    return torch.cat([x, y], dim=1)
