"""Convlution layer and Linear layer which support equalized learning rate. """
import torch
import torch.nn.functional as F


def get_weight(shape, gain=1., use_wscale=False, lrmul=1.0):
    """Surpports equalized learning rate for weights updating. """
    fan_in = torch.prod(torch.FloatTensor(list(shape[1:])))
    he_std = gain / torch.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = torch.FloatTensor([lrmul])

    return init_std, runtime_coef


def _init_parameters(_m, init_std):
    torch.nn.init.normal_(_m.weight, 0., init_std)
    if _m.bias is not None:
        torch.nn.init.constant_(_m.bias, 0.)


def _blur_w(weight, factor=1.):
    assert weight.ndim == 4 and all(weight.shape)

    _w = F.pad(weight, (1, 1, 1, 1), mode='constant', value=0.)
    _w = _w[..., 1:, 1:] + _w[..., :-1, 1:] + _w[..., 1:, :-1] + _w[..., :-1, :-1]

    return _w * factor


class Linear(torch.nn.Linear):
    """Runtime weight scaled Linear. """
    def __init__(self, in_features, out_features, bias=False, gain=1., use_wscale=False, lrmul=1.):
        super(Linear, self).__init__(in_features, out_features, bias)

        init_std, runtime_coef = get_weight(self.weight.shape, gain, use_wscale, lrmul)
        _init_parameters(self, init_std)

        self.register_buffer("runtime_coef", runtime_coef)

    def forward(self, input):
        return F.linear(input, self.weight * self.runtime_coef, self.bias)


class Conv2d(torch.nn.Conv2d):
    """Runtime weight scaled Conv2d. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_mode='zeros',
                 gain=1., use_wscale=False, lrmul=1.):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)

        init_std, runtime_coef = get_weight(self.weight.shape, gain, use_wscale, lrmul)
        _init_parameters(self, init_std)

        self.register_buffer("runtime_coef", runtime_coef)

    def forward(self, input, blur=False):
        weight = self.weight * self.runtime_coef
        if blur:
            weight = _blur_w(weight, 0.25)

        return self.conv2d_forward(input, weight)


class ConvTranspose2d(torch.nn.ConvTranspose2d):
    """Runtime weight scaled ConvTranspose2d. """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=False, dilation=1,
                 padding_mode='zeros', gain=1., use_wscale=False, lrmul=1.):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, output_padding, groups, bias,
                                              dilation, padding_mode)

        init_std, runtime_coef = get_weight(self.weight.shape, gain, use_wscale, lrmul)
        _init_parameters(self, init_std)

        self.register_buffer("runtime_coef", runtime_coef)

    def forward(self, input, blur=False, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride,
                                              self.padding, self.kernel_size)

        weight = self.weight * self.runtime_coef
        if blur:
            weight = _blur_w(weight, 1.)

        return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

__all__ = ["Linear", "Conv2d", "ConvTranspose2d"]
