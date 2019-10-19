"""Network architectures used in StyleGAN paper. """
import torch


def _init_weight(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def _upscale2d(x, factor=2):
    assert x.ndim == 4 and all(x.shape)
    assert isinstance(factor, int) and factor >= 1

    if factor == 1: return x

    s = x.shape
    x = x.reshape(*s[:3], 1, s[3], 1)
    x = x.expand(*s[:3], factor, s[3], factor)
    x = x.reshape(*s[:2], -1, s[3] * factor)

    return x


def _downscale2d(x, factor=2, f=(1, 2, 1), normalize=True, flip=False):
    assert x.ndim == 4 and all(x.shape)
    assert isinstance(factor, int) and factor >= 1

    if factor == 2:
        padding, rem = divmod(len(f) - factor, 2)
        padding += rem == 1

        f = torch.FloatTensor(f).to(x.device)
        if f.ndim == 1: f = f * f.reshape(-1, 1)
        assert f.ndim == 2
        if normalize: f /= f.sum()
        if flip: f = f[::-1, ::-1]
        f = f.reshape(1, 1, *f.shape)
        f = f.expand(x.size(1), 1, *f.shape[2:])

        return torch.nn.functional.conv2d(x, f, stride=factor, padding=padding, groups=x.size(1))

    if factor == 1: return x

    return torch.nn.functional.avg_pool2d(x, factor, factor)


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
    y = torch.mean(y, dim=0, keepdim=True) # [1MncHW] Subtract mean over group.
    y = torch.mean(y ** 2, dim=0)          # [MncHW]  Calc variance over group.
    y = torch.sqrt(y + 1e-8)               # [MncHW]  Calc stddev over group.
    y = torch.mean(y, dim=(2, 3, 4), keepdim=True) # [Mn111] Take average over fmaps and pixels.
    y = torch.mean(y, dim=2)                       # [Mn11]  Split channels into c channel groups.
    y = y.repeat(group_size, num_new_features, *s[2:]) # [NnHW] Replicate over group and pixels.

    return torch.cat([x, y], dim=1)


class D_basic(torch.nn.Module):
    """Discriminator used in the StyleGAN paper. """
    def __init__(self, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 stage=2, label_size=0, mbstd_group_size=4, mbstd_num_features=1, blur_filter=(1, 2, 1)):
        super(D_basic, self).__init__()

        resolution_log2 = torch.log2(torch.FloatTensor([resolution])).item()
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.resolution_log2 = int(resolution_log2)
        self._nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)
        self._downscale = lambda x: _downscale2d(x, f=blur_filter)

        self.stage = 1
        self.num_channels = num_channels

        self.final = _OutputD(self._nf(1), label_size, mbstd_group_size, mbstd_num_features)
        self.sieving = torch.nn.ModuleDict()

        self.fromrgb = torch.nn.ModuleDict({
            'stage_0': torch.nn.Conv2d(num_channels, self._nf(1), 1, 1, 0),
        })

        for _ in range(1, stage):
            self.__grow()

    def forward(self, image, alpha=1, stage=None, label=None):
        stage = [stage, self.stage][stage is None]
        if self.training and stage > self.stage:
            for _ in range(self.stage, stage):
                self.__grow()
            if next(self.final.parameters()).is_cuda: self.cuda()
        assert 1 <= stage <= self.stage, "stage exceeding!"

        x = self.fromrgb[f'stage_{stage - 1}'](image)
        if stage >= 2:
            _x = self.fromrgb[f'stage_{stage - 2}'](self._downscale(image))
            x = torch.lerp(_x, self.sieving[f'stage_{stage - 1}'](x), alpha)
            for i in range(stage - 2, 0, -1):
                x = self.sieving[f'stage_{i}'](x)

        return self.final(x, label)

    def __grow(self):
        """Supports progressive grwing. """
        self.sieving.update({
            'stage_{}'.format(self.stage):
            DownScaleConv2d(self._nf(self.stage + 1), self._nf(self.stage)),
        })
        self.fromrgb.update({
            'stage_{}'.format(self.stage):
            torch.nn.Conv2d(self.num_channels, self._nf(self.stage + 1), 1, 1, 0),
        })
        self.stage += 1


class _OutputD(torch.nn.Module):
    def __init__(self, fmap_in, label_size=0, mbstd_group_size=4, mbstd_num_features=1):
        super(_OutputD, self).__init__()

        self.label_size = label_size
        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(fmap_in + mbstd_num_features, fmap_in, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(fmap_in, fmap_in, 4, 1, 0),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(fmap_in, max(label_size, 1), 1, 1, 0),
        )

        self.mbstd_group_size = mbstd_group_size
        if mbstd_group_size > 1:
            self.mbstd = lambda x: minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)

        self.apply(_init_weight)

    def forward(self, x, label=None):
        if self.mbstd_group_size > 1:
            x = self.mbstd(x)
        x = self.sub_module(x)
        x = x.reshape(x.size(0), -1)
        if self.label_size:
            assert label is not None and label.size(1) == self.label_size
            x = torch.sum(x * label, dim=1, keepdim=True)

        return x


class DownScaleConv2d(torch.nn.Module):
    """Building blocks for dicriminator. """
    def __init__(self, fmap_in, fmap_out):
        super(DownScaleConv2d, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(fmap_in, fmap_in, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(fmap_in, fmap_out, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.apply(_init_weight)

    def forward(self, x):
        return self.sub_module(x)


class G_style(torch.nn.Module):
    """Style-based generator used in the StyleGAN paper.
       Composed of two sub-networks (GMapping and GSynthesis) that are defined below.
    """
    def __init__(self, stage=2, dlatent_avg_beta=0.995, dlatent_size=512, **kwargs):
        super(G_style, self).__init__()

        self.mapping = GMapping(dlatent_size=dlatent_size, **kwargs)
        self.synthesis = GSynthesis(stage=stage, dlatent_size=dlatent_size, **kwargs)

        self.dlatent_avg_beta = dlatent_avg_beta
        self.register_buffer('dlatent_avg', torch.zeros(dlatent_size))

    def forward(self, latent, alpha=1, stage=None, label=None, mixing_prob=0.9, beta=0.7, cutoff=8, **kwargs):
        """Args:
               mixing_prob: probability of mixing styles during training. None = disable.
               beta: style strength multiplier for the truncation trick. None = disable.
               cutoff: number of layers for which to apply the truncation trick. None = disable.
        """
        if self.training or (beta is not None and beta == 1):
            beta = None
        if self.training or (cutoff is not None and cutoff <= 0):
            cutoff = None
        if not self.training or (mixing_prob is not None and  mixing_prob <= 0):
            mixing_prob = None
        if not self.training or self.dlatent_avg_beta == 1:
            self.dlatent_avg_beta = None
        if stage is None or (stage is not None and stage <= 0):
            stage = self.synthesis.stage

        dlatents = self._dlatents(stage, latent, label, mixing_prob, beta, cutoff, **kwargs)
        noises = self._noises(stage, latent.size(0), latent.device, **kwargs)

        return self.synthesis(alpha, stage, dlatents, noises, **kwargs)

    def _dlatents(self, stage, latent, label, mixing_prob, beta, cutoff, latent2=None, section=None, **kwargs):
        """Provides intermediate latent space as demand from input latent (with label.)

           Args:
               latent2: use styles mapped from this input latent besides the single one. None = disable.
               section: stages to apply the additional styles. None = disable.
        """
        assert stage + 1 <= self.synthesis.resolution_log2, "stage exceeding!"
        num_layers = 2 * stage
        device = latent.device

        dlatent = self.mapping(latent, label, **kwargs)

        # Update moving average of W.
        if self.dlatent_avg_beta is not None:
            batch_avg = torch.mean(dlatent, dim=0)
            self.dlatent_avg = torch.lerp(batch_avg, self.dlatent_avg, self.dlatent_avg_beta)

        dlatents = torch.unsqueeze(dlatent, dim=0).repeat_interleave(num_layers, dim=0)
        layer_idx = torch.arange(num_layers, device=device).reshape(-1, 1, 1)

        # Perform style mixing regularization.
        if mixing_prob is not None:
            _latent2 = torch.randn_like(latent)
            _dlatent2 = self.mapping(_latent2, label, **kwargs)
            mix_cutoff = torch.where(torch.rand(1) < mixing_prob, torch.randint(num_layers, (1,)),
                                     torch.tensor([num_layers])).to(device)
            dlatents = torch.where(layer_idx < mix_cutoff, dlatent, _dlatent2)

        # Multi-styles.
        if latent2 is not None and section is not None:
            dlatent2 = self.mapping(latent2, label, **kwargs)
            dlatents = torch.where(sum([layer_idx == i for i in section]), dlatent2, dlatent)

        # Apply truncation trick.
        if beta is not None and cutoff is not None:
            ones = torch.ones_like(layer_idx, dtype=torch.float)
            coefs = torch.where(layer_idx < cutoff, beta * ones, ones)
            dlatents = torch.lerp(self.dlatent_avg, dlatents, coefs)

        return dlatents.reshape(-1, 2, *dlatents.shape[1:])

    def _noises(self, stage, batch_size, device, noise_section=None, **_kwargs):
        """Provides noises to synthesis process.

           Args:
               noise_section: stages to apply noises if specified. None means apply noises to all stages.
        """
        assert stage + 1 <= self.synthesis.resolution_log2, "stage exceeding!"
        if noise_section is None: noise_section = range(1, stage + 1)

        noises = []
        resolution = 4
        for i in range(1, stage + 1):
            shape = (2, batch_size, 1, resolution, resolution)
            noise = torch.randn(shape, device=device) if i in noise_section else torch.zeros(shape, device=device)
            noises.append(noise)
            resolution *= 2

        return noises


class GMapping(torch.nn.Module):
    """Mapping network used in StyleGAN. """
    def __init__(self, latent_size=512, label_size=0, dlatent_size=512, mapping_layers=8, mapping_fmaps=512, **_kwargs):
        super(GMapping, self).__init__()

        self.label_size = label_size

        if label_size:
            self.label_embed = torch.nn.Linear(label_size, latent_size, bias=False)

        self.sub_module = self._make_layers(latent_size * (1 + (label_size > 0)),
                                            dlatent_size, mapping_layers, mapping_fmaps)

        self.apply(_init_weight)

    def forward(self, latent, label=None, normalize_latent=True, **_kwargs):
        if self.label_size > 0:
            assert label is not None, "labels needed!"
            label = self.label_embed(label)
        else:
            label = torch.empty(latent.size(0), 0, device=latent.device)

        _z = torch.cat([latent, label], dim=1)
        if normalize_latent:
            _z = pixel_norm(_z)

        _w = self.sub_module(_z)
        return _w

    @staticmethod
    def _make_layers(in_features, out_features, mapping_layers, mapping_fmaps):
        layers = []

        _fmaps = [in_features] + [mapping_fmaps] * (mapping_layers - 1) + [out_features]
        for _in, _out in zip(_fmaps[:-1], _fmaps[1:]):
            layers.extend([torch.nn.Linear(_in, _out), torch.nn.LeakyReLU(0.2, inplace=True)])

        return torch.nn.Sequential(*layers)


class GSynthesis(torch.nn.Module):
    """Synthesis network used in StyleGAN. """
    def __init__(self, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512, stage=None,
                 ilatent_size=0, const_input=True, dlatent_size=512, use_styles=True, use_noise=True, **_kwargs):
        super(GSynthesis, self).__init__()

        resolution_log2 = torch.log2(torch.FloatTensor([resolution])).item()
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.resolution_log2 = int(resolution_log2)
        self._nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)

        self.stage = 1
        self.num_channels = num_channels
        self._kwargs = dict(dlatent_size=dlatent_size, use_styles=use_styles, use_noise=use_noise)

        self.newborn = _InputG(self._nf(1), ilatent_size, const_input, **self._kwargs)
        self.growing = torch.nn.ModuleDict()

        self.torgb = torch.nn.ModuleDict({'stage_0': torch.nn.Conv2d(self._nf(1), num_channels, 1, 1, 0)})

        for _ in range(1, [stage, self.resolution_log2 - 1][stage is None]):
            self.__grow()

    def forward(self, alpha=1, stage=None, dlatents=None, noises=None, ilatent=None, **_kwargs):
        stage = [stage, self.stage][stage is None]
        if self.training and stage > self.stage:
            for _ in range(self.stage, stage):
                self.__grow()
            if next(self.newborn.parameters()).is_cuda: self.cuda()
        assert 1 <= stage <= self.stage, "stage excceeding!"

        if dlatents is None: dlatents = [[None, None]] * self.stage
        if noises is None: noises = [[None, None]] * self.stage

        _x = self.newborn(dlatents[0], noises[0], ilatent)
        if stage == 1: return self.torgb['stage_0'](_x)

        _i = 1
        while _i < stage - 1:
            _x = self.growing[f'stage_{_i}'](_x, dlatents[_i], noises[_i])
            _i += 1
        _rgb = self.torgb[f'stage_{_i - 1}'](_x)

        _x = self.growing[f'stage_{_i}'](_x, dlatents[_i], noises[_i])
        rgb = self.torgb[f'stage_{_i}'](_x)

        return torch.lerp(_upscale2d(_rgb), rgb, alpha)

    def __grow(self):
        """Supports progressive growing. """
        assert self.stage + 2 <= self.resolution_log2, "stage exceeding upper limit!"

        self.growing.update({
            'stage_{}'.format(self.stage):
            UpScaleConv2d(self._nf(self.stage), self._nf(self.stage + 1), **self._kwargs),
        })
        self.torgb.update({
            'stage_{}'.format(self.stage):
            torch.nn.Conv2d(self._nf(self.stage + 1), self.num_channels, 1, 1, 0),
        })
        self.stage += 1

class _InputG(torch.nn.Module):

    def __init__(self, num_features, ilatent_size=0, const_input=True, **kwargs):
        super(_InputG, self).__init__()

        self.const_input = const_input
        self.num_features = num_features

        if const_input:
            self.layer_0 = torch.nn.Parameter(torch.ones(1, num_features, 4, 4))
        else:
            self.layer_0 = torch.nn.Linear(ilatent_size, num_features * 16, bias=False)
        self.epilog_0 = EpilogueLayer(num_features, **kwargs)
        self.layer_1 = torch.nn.Conv2d(num_features, num_features, 3, 1, 1, bias=False)
        self.epilog_1 = EpilogueLayer(num_features, **kwargs)

        self.apply(_init_weight)

    def forward(self, dlatents=[None, None], noises=[None, None], ilatent=None):
        if self.const_input:
            assert dlatents[0] is not None, "dlatent needed!"
            x = torch.repeat_interleave(self.layer_0, dlatents[0].size(0), dim=0)
        else:
            assert ilatent is not None, "ilatent needed!"
            x = self.layer_0(ilatent).reshape(-1, self.num_features, 4, 4)
        x = self.epilog_0(x, dlatents[0], noises[0])
        x = self.layer_1(x)
        x = self.epilog_1(x, dlatents[1], noises[1])

        return x


class UpScaleConv2d(torch.nn.Module):
    """Building blocks for generator. """
    def __init__(self, fmap_in, fmap_out, **kwargs):
        super(UpScaleConv2d, self).__init__()

        self.layer_0 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(fmap_in, fmap_out, 3, 1, 1, bias=False),
        )
        self.epilog_0 = EpilogueLayer(fmap_out, **kwargs)
        self.layer_1 = torch.nn.Conv2d(fmap_out, fmap_out, 3, 1, 1, bias=False)
        self.epilog_1 = EpilogueLayer(fmap_out, **kwargs)

        self.apply(_init_weight)

    def forward(self, x, dlatents=[None, None], noises=[None, None]):
        x = self.layer_0(x)
        x = self.epilog_0(x, dlatents[0], noises[0])
        x = self.layer_1(x)
        x = self.epilog_1(x, dlatents[1], noises[1])

        return x


class EpilogueLayer(torch.nn.Module):
    """Things to do at the end of each layer. """
    def __init__(self, num_features, dlatent_size=512, use_styles=True, use_noise=True, **_kwargs):
        super(EpilogueLayer, self).__init__()

        self.use_styles = use_styles
        self.use_noise = use_noise

        if use_noise:
            self.apply_noise = ApplyNoise(num_features)

        self.bias = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.act = torch.nn.LeakyReLU(0.2, inplace=True)

        if use_styles:
            self.style_mod = StyleMod(num_features, dlatent_size)

    def forward(self, x, dlatent=None, noise=None):
        if self.use_noise:
            assert noise is not None, "noise needed!"
            x = self.apply_noise(x, noise)
        x += self.bias
        x = self.act(x)
        if self.use_styles:
            assert dlatent is not None, "dlatent needed!"
            x = self.style_mod(x, dlatent)

        return x


class ApplyNoise(torch.nn.Module):
    """Noise input for corresponding convolution layer. """
    def __init__(self, num_features):
        super(ApplyNoise, self).__init__()

        self.scaling_factors = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, features, noise):
        return features + noise * self.scaling_factors


class StyleMod(torch.nn.Module):
    """Style modulation, which controls the generator through AdaIN at corresponding convolution layer. """
    def __init__(self, num_features, dlatent_size):
        super(StyleMod, self).__init__()

        self.affine_transform = torch.nn.Linear(dlatent_size, 2 * num_features)

    def forward(self, features, dlatent):
        styles = self.affine_transform(dlatent)
        contents = instance_norm(features)

        styles = styles.reshape(-1, 2, contents.size(1), 1, 1)
        features = contents * (styles[:, 0] + 1) + styles[:, 1]

        return features
