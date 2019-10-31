"""Upscale layer and Downscale layer. """
import torch
from .eqlr import Conv2d, ConvTranspose2d


#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.

def blur2d(x, f=[1, 2, 1], normalize: bool = True, flip: bool = False, stride: int = 1):
    assert x.ndim == 4 and all(x.shape)
    assert stride >= 1

    padding, rem = divmod(len(f) - stride, 2)
    padding += rem == 1

    f = torch.FloatTensor(f).to(x.device)
    if f.ndim == 1: f = f * f.reshape(-1, 1)
    assert f.ndim == 2

    if normalize: f /= f.sum()
    if flip: f = f[::-1, ::-1]

    f = f.reshape(1, 1, *f.shape)
    f = f.expand(x.size(1), 1, *f.shape[2:])

    return torch.nn.functional.conv2d(x, f, stride=stride, padding=padding, groups=x.size(1))


def upscale2d(x, gain=1., factor: int = 2):
    assert x.ndim == 4 and all(x.shape)
    assert factor >= 1

    if gain != 1: x *= gain

    if factor == 1: return x

    s = x.shape
    x = x.reshape(*s[:3], 1, s[3], 1)
    x = x.expand(*s[:3], factor, s[3], factor)
    x = x.reshape(*s[:2], -1, s[3] * factor)

    return x


def downscale2d(x, gain=1., factor: int = 2):
    assert x.ndim == 4 and all(x.shape)
    assert isinstance(factor, int) and factor >= 1

    if factor == 2:
        f = [(gain ** 0.5) / factor] * factor
        return blur2d(x, f=f, normalize=False, stride=factor)

    if gain != 1: x *= gain

    if factor == 1: return x

    return torch.nn.functional.avg_pool2d(x, factor, factor)


#----------------------------------------------------------------------------
# Fused convolution + scaling.
# Faster and uses less memory than performing the operations separately.

class UpscaleConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, fused_scale=False,
                 blur_filter=[1, 2, 1], **kwargs):
        assert kernel_size >= 1 and kernel_size % 2 == 1
        super(UpscaleConv2d, self).__init__()

        stride = [1, 2][fused_scale]
        padding, rem = divmod(kernel_size - stride, 2)
        padding += rem == 1

        if not fused_scale:
            self.sub_module = Conv2d(in_channels, out_channels, kernel_size,
                                     stride, padding, **kwargs)
        else:
            self.sub_module = ConvTranspose2d(in_channels, out_channels, kernel_size,
                                              stride, padding, **kwargs)

        self.blur = lambda x: blur2d(x, blur_filter) if blur_filter else x
        self.register_buffer("fused_scale", torch.BoolTensor([fused_scale]))

    def forward(self, input):
        if not self.fused_scale:
            x = self.sub_module(upscale2d(input))
        else:
            x = self.sub_module(input, blur=True)

        return self.blur(x)


class ConvDownscale2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, fused_scale=False,
                 blur_filter=[1, 2, 1], **kwargs):
        assert kernel_size >= 1 and kernel_size % 2 == 1
        super(ConvDownscale2d, self).__init__()

        stride = [1, 2][fused_scale]
        padding, rem = divmod(kernel_size - stride, 2)
        padding += rem == 1

        self.sub_module = Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, **kwargs)

        self.blur = lambda x: blur2d(x, blur_filter) if blur_filter else x
        self.register_buffer("fused_scale", torch.BoolTensor([fused_scale]))

    def forward(self, input):
        x = self.blur(input)
        if not self.fused_scale:
            x = downscale2d(self.sub_module(input))
        else:
            x = self.sub_module(input, blur=True)

        return x
