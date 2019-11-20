"""Network architectures used in StyleGAN paper. """
from math import sqrt, log2
import torch

from ganlib.eqlr import Linear, Conv2d
from ganlib.scale import upscale2d, downscale2d, UpscaleConv2d, ConvDownscale2d
from ganlib.normalize import pixel_norm, instance_norm, minibatch_stddev_layer


_ACTIVATE = {'relu': (torch.nn.ReLU(), sqrt(2)), 'lrelu': (torch.nn.LeakyReLU(0.2), sqrt(2))}


class D_basic(torch.nn.Module):
    """Discriminator used in the StyleGAN paper. """
    def __init__(self, num_channels=3, resolution=32, label_size=0, fmap_base=8192,
                 fmap_decay=1.0, fmap_max=512, nonlinearity='lrelu', use_wscale=True,
                 mbstd_group_size=4, mbstd_num_features=1, fused_scale='auto',
                 blur_filter=[1, 2, 1], **_kwargs):
        super(D_basic, self).__init__()

        self.res = int(log2(resolution))
        assert resolution == 2 ** self.res and resolution >= 4
        nf = lambda res: min(int(fmap_base / (2 ** (res * fmap_decay))), fmap_max)
        act, gain = _ACTIVATE[nonlinearity]

        self.trunk = torch.nn.ModuleList([
            _EpilogLayer(nf(1), nf(0), act, label_size, mbstd_group_size,
                         mbstd_num_features, gain, use_wscale)
        ])
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(
                Conv2d(num_channels, nf(1), 1, 1, 0, bias=True, gain=gain, use_wscale=use_wscale),
                act,
            )
        ])

        self._block = lambda res: self.trunk.append(
            _DownscaleBlock(nf(res - 1), nf(res - 2), act, gain, use_wscale,
                            res >= 7 if fused_scale == 'auto' else fused_scale, blur_filter)
        )
        self._fromrgb = lambda res: self.branches.append(
            torch.nn.Sequential(
                Conv2d(num_channels, nf(res - 1), 1, 1, 0, bias=True,
                       gain=gain, use_wscale=use_wscale),
                act,
            )
        )

        self.register_buffer("use_label", torch.BoolTensor([label_size > 0]))

        for res in range(3, self.res + 1):
            self._block(res)
            self._fromrgb(res)

    def forward(self, images, res=None, alpha=0., labels=None):
        res = self.res if res is None else res
        assert isinstance(res, int) and res >= 3
        if not self.training:
            assert res <= self.res, f"resolution too large, should be less than {2 ** self.res}"

        def grow(res, lod):
            if self.training and res > self.res:
                self.res += 1
                self._block(self.res)
                self._fromrgb(self.res)
                if next(self.trunk[0].parameters()).is_cuda:
                    self.trunk[-1].cuda()
                    self.branches[-1].cuda()

            x = lambda: self.branches[res - 2](downscale2d(images, factor=2**lod))
            if lod > 0 and lod > alpha:
                x = lambda: grow(res + 1, lod - 1)
            x = self.trunk[res - 2](x())
            y = lambda: x
            if res > 2 and alpha > lod:
                y = lambda: torch.lerp(x, self.branches[res - 3]\
                                  (downscale2d(images, factor=2**(lod + 1))), alpha - lod)

            return y()

        scores_out = grow(2, res - 2)
        scores_out = scores_out.reshape(scores_out.size(0), -1)

        # Label conditioning from "Which Training Methods for GANs do actually Converge?"
        if self.use_label:
            assert labels is not None, "labels needed!"
            scores_out = torch.sum(scores_out * labels, dim=1, keepdim=True)

        return scores_out


class _EpilogLayer(torch.nn.Module):

    def __init__(self, fmap_in, fmap_out, activate, label_size=0, mbstd_group_size=4,
                 mbstd_num_features=1, gain=1., use_wscale=False):
        super(_EpilogLayer, self).__init__()

        self.mbstd = lambda x: minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features) \
            if mbstd_group_size > 1 else x

        self.sub_module = torch.nn.Sequential(
            Conv2d(fmap_in + [0, mbstd_num_features][mbstd_group_size > 1], fmap_in,
                   3, 1, 1, bias=True, gain=gain, use_wscale=use_wscale),
            activate,
            Conv2d(fmap_in, fmap_out, 4, 1, 0, bias=True, gain=gain, use_wscale=use_wscale),
            activate,
            Conv2d(fmap_out, max(label_size, 1), 1, 1, 0, bias=True, gain=1., use_wscale=use_wscale)
        )

    def forward(self, x):
        x = self.mbstd(x)
        x = self.sub_module(x)

        return x


class _DownscaleBlock(torch.nn.Module):

    def __init__(self, fmap_in, fmap_out, activate, gain=1., use_wscale=True,
                 fused_scale=False, blur_filter=[1, 2, 1]):
        super(_DownscaleBlock, self).__init__()

        self.sub_module = torch.nn.Sequential(
            Conv2d(fmap_in, fmap_in, 3, 1, 1, bias=True, gain=gain, use_wscale=use_wscale),
            activate,
            ConvDownscale2d(fmap_in, fmap_out, 3, fused_scale, blur_filter,
                            gain=gain, use_wscale=use_wscale),
            Bias(fmap_out),
            activate,
        )

    def forward(self, x):
        return self.sub_module(x)


class G_style(torch.nn.Module):
    """Style-based generator used in the StyleGAN paper.
       Composed of two sub-networks (GMapping and GSynthesis) that are defined below.
    """
    def __init__(self, dlatent_avg_beta=0.995, dlatent_size=512, **kwargs):
        super(G_style, self).__init__()

        self.mapping = GMapping(dlatent_size=dlatent_size, **kwargs)
        self.synthesis = GSynthesis(dlatent_size=dlatent_size, **kwargs)

        self.dlatent_avg_beta = dlatent_avg_beta
        self.register_buffer('dlatent_avg', torch.zeros(dlatent_size))

    def forward(self, latent, res=None, alpha=0., label=None, style_mixing_prob=0.9,
                truncation_psi=0.7, truncation_cutoff=8, randomize_noise=True):
        """Args:
               style_mixing_prob: probability of mixing styles during training. None = disable.
               truncation_psi: style strength multiplier for the truncation trick. None = disable.
               truncation_cutoff: number of layers for which to apply the truncation trick.
                                  None = disable.
        """
        if self.training or (truncation_psi is not None and truncation_psi == 1):
            truncation_psi = None
        if self.training or (truncation_cutoff is not None and truncation_cutoff <= 0):
            truncation_cutoff = None
        if not self.training or (style_mixing_prob is not None and  style_mixing_prob <= 0):
            style_mixing_prob = None
        if not self.training or self.dlatent_avg_beta == 1:
            self.dlatent_avg_beta = None
        if res is None:
            res = self.synthesis.res

        dlatent = self.mapping(latent, label)

        # Update moving average of W.
        if self.dlatent_avg_beta is not None:
            batch_avg = torch.mean(dlatent, dim=0)
            self.dlatent_avg = torch.lerp(batch_avg, self.dlatent_avg, self.dlatent_avg_beta)

        num_layers = 2 * (res - 1)
        device = latent.device

        dlatents = torch.unsqueeze(dlatent, dim=0).repeat_interleave(num_layers, dim=0)
        layer_idx = torch.arange(num_layers, device=device).reshape(-1, 1, 1)

        # Perform style mixing regularization.
        if style_mixing_prob is not None:
            latent2 = torch.randn_like(latent)
            dlatent2 = self.mapping(latent2, label)
            mix_cutoff = torch.where(torch.rand(1) < style_mixing_prob,
                                     torch.randint(1, num_layers, (1,)),
                                     torch.tensor([num_layers])).item()
            dlatents = torch.where(layer_idx < mix_cutoff, dlatent, dlatent2)

        # Apply truncation trick.
        if truncation_psi is not None and truncation_cutoff is not None:
            ones = torch.ones_like(layer_idx, dtype=torch.float)
            coefs = torch.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = torch.lerp(self.dlatent_avg, dlatents, coefs)

        return self.synthesis(res, alpha, dlatents, randomize_noise)


class GMapping(torch.nn.Module):
    """Mapping network used in StyleGAN. """
    def __init__(self, latent_size=512, label_size=0, dlatent_size=512, mapping_layers=8,
                 mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **_kwargs):
        super(GMapping, self).__init__()

        act, gain = _ACTIVATE[mapping_nonlinearity]

        if label_size > 0:
            self.register_parameter("embed",
                                    torch.nn.Parameter(torch.randn(label_size, latent_size)))

        latent_size *= 1 + (label_size > 0)
        _fmaps = [latent_size] + [mapping_fmaps] * (mapping_layers - 1) + [dlatent_size]
        layers = []
        for _in, _out in zip(_fmaps[:-1], _fmaps[1:]):
            layers.extend([Linear(_in, _out, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul),
                           Bias(_out, lrmul=mapping_lrmul), act])
        self.mapping = torch.nn.Sequential(*layers)

        self.register_buffer("action", torch.BoolTensor([label_size > 0, normalize_latents]))

    def forward(self, latent, label=None):
        if self.action[0] > 0:
            assert label is not None, "label needed!"
            embed = torch.matmul(label, self.embed)
            latent = torch.cat([latent, embed], dim=1)

        if self.action[1]:
            _z = pixel_norm(latent)
        _w = self.mapping(_z)

        return _w


class GSynthesis(torch.nn.Module):
    """Synthesis network used in StyleGAN. """
    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024, const_input=True,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512, use_noise=True,
                 use_pixel_norm=False, use_instance_norm=True, use_styles=True,
                 nonlinearity='lrelu', use_wscale=True, fused_scale='auto',
                 blur_filter=[1, 2, 1], **_kwargs):
        super(GSynthesis, self).__init__()

        self.res = int(log2(resolution))
        assert resolution == 2 ** self.res and resolution >= 4
        nf = lambda res: min(int(fmap_base / (2 ** (res * fmap_decay))), fmap_max)
        act, gain = _ACTIVATE[nonlinearity]

        self.trunk = torch.nn.ModuleList([
            _EarlyLayer(dlatent_size, nf(1), 4, act, const_input, gain, use_wscale,
                        use_noise=use_noise, use_pixel_norm=use_pixel_norm,
                        use_instance_norm=use_instance_norm, use_styles=use_styles),
        ])
        self.leaves = torch.nn.ModuleList([
            Conv2d(nf(1), num_channels, 1, 1, 0, bias=True, gain=1., use_wscale=use_wscale),
        ])

        self._block = lambda res: self.trunk.append(
            _UpscaleBlock(nf(res - 2), nf(res - 1), dlatent_size, act, gain, use_wscale,
                          res >= 7 if fused_scale == 'auto' else fused_scale,
                          blur_filter, use_noise=use_noise, use_pixel_norm=use_pixel_norm,
                          use_instance_norm=use_instance_norm, use_styles=use_styles),
        )
        self._torgb = lambda res: self.leaves.append(
            Conv2d(nf(res - 1), num_channels, 1, 1, 0, bias=True, gain=1., use_wscale=use_wscale)
        )

        self.register_buffer("use_noise", torch.BoolTensor([use_noise]))

        for res in range(3, self.res + 1):
            self._block(res)
            self._torgb(res)

    def forward(self, res=None, alpha=0., dlatent=None, randomize_noise=True):
        res = self.res if res is None else res
        assert isinstance(res, int) and res >= 3
        if not self.training:
            assert res <= self.res, f"resolution too large, should be less than {2 ** self.res}"
        is_cuda = next(self.trunk[0].parameters()).is_cuda

        dlatent = [None, None] * (res - 1) if dlatent is None else dlatent
        noise = []
        if self.use_noise:
            for layer_idx in range(res * 2 - 2):
                _res = layer_idx // 2 + 2
                noise.append(torch.randn(1, self.use_noise, 2**_res, 2**_res,
                                         device=['cpu', 'cuda'][is_cuda]))

        _x = self.trunk[0](dlatent[:2], noise[:2], randomize_noise)

        def grow(x, res, lod):
            if self.training and res > self.res:
                self.res += 1
                self._block(self.res)
                self._torgb(self.res)
                if is_cuda:
                    self.trunk[-1].cuda()
                    self.leaves[-1].cuda()

            _y = self.trunk[res - 2](x, dlatent[2*(res - 2):2*(res - 1)],
                                     noise[2*(res - 2):2*(res - 1)], randomize_noise)
            img = lambda: upscale2d(self.leaves[res - 2](_y), factor=2 ** lod)
            if alpha > lod:
                img = lambda: upscale2d(
                    torch.lerp(self.leaves[res - 2](_y), upscale2d(self.leaves[res - 3](x)),
                               alpha - lod),
                    factor=2 ** lod)
            if lod > 0 and alpha < lod:
                img = lambda: grow(_y, res + 1, lod - 1)

            return img()

        images_out = grow(_x, 3, res - 3)

        return images_out


class _EarlyLayer(torch.nn.Module):

    def __init__(self, dlatent_size, fmap_out, size, activate, const_input=True, gain=1.,
                 use_wscale=False, **kwargs):
        super(_EarlyLayer, self).__init__()

        if const_input:
            self.register_buffer("const", torch.nn.Parameter(torch.ones(1, fmap_out, size, size)))
        else:
            self.embed = Linear(dlatent_size, fmap_out * size ** 2, gain=gain/4,
                                use_wscale=use_wscale)
        self.sub_module = torch.nn.ModuleList([
            EpilogueLayer(fmap_out, dlatent_size, activate, use_wscale=use_wscale, **kwargs),
            Conv2d(fmap_out, fmap_out, 3, 1, 1, gain=gain, use_wscale=use_wscale),
            EpilogueLayer(fmap_out, dlatent_size, activate, use_wscale=use_wscale, **kwargs),
        ])

        self.const_input = const_input
        self.size = size

    def forward(self, dlatent, noise_var, randomize_noise):
        if self.const_input:
            x = self.const.repeat_interleave(dlatent[0].size(0), dim=0)
        else:
            x = self.embed(dlatent[0])
            x = x.reshape(x.shape[0], -1, self.size, self.size)

        dlatent = [None] * 2 if len(dlatent) == 1 else dlatent

        x = self.sub_module[0](x, dlatent[0], noise_var[0], randomize_noise)
        x = self.sub_module[2](self.sub_module[1](x), dlatent[1], noise_var[1], randomize_noise)

        return x


class _UpscaleBlock(torch.nn.Module):

    def __init__(self, fmap_in, fmap_out, dlatent_size, activate, gain=1., use_wscale=True,
                 fused_scale=False, blur_filter=[1, 2, 1], **kwargs):
        super(_UpscaleBlock, self).__init__()

        self.sub_module = torch.nn.ModuleList([
            UpscaleConv2d(fmap_in, fmap_out, 3, fused_scale, blur_filter,
                          gain=gain, use_wscale=use_wscale),
            EpilogueLayer(fmap_out, dlatent_size, activate, use_wscale=use_wscale, **kwargs),
            Conv2d(fmap_out, fmap_out, 3, 1, 1, gain=gain, use_wscale=use_wscale),
            EpilogueLayer(fmap_out, dlatent_size, activate, use_wscale=use_wscale, **kwargs),
        ])

    def forward(self, x, dlatent, noise_var, randomize_noise):
        dlatent = [None] * 2 if len(dlatent) == 0 else dlatent

        x = self.sub_module[1](self.sub_module[0](x), dlatent[0], noise_var[0], randomize_noise)
        x = self.sub_module[3](self.sub_module[2](x), dlatent[1], noise_var[1], randomize_noise)

        return x


class EpilogueLayer(torch.nn.Module):
    """Things to do at the end of each layer. """
    def __init__(self, num_features, dlatent_size, activate, use_noise=True, use_pixel_norm=False,
                 use_instance_norm=True, use_styles=True, use_wscale=False):
        super(EpilogueLayer, self).__init__()

        if use_noise:
            self.apply_noise = ApplyNoise(num_features)

        self.bias = Bias(num_features)
        self.act = activate

        if use_styles:
            self.style_mod = StyleMod(num_features, dlatent_size, use_wscale=use_wscale)

        self.register_buffer("action", torch.BoolTensor([use_noise, use_pixel_norm,
                                                         use_instance_norm, use_styles]))

    def forward(self, x, dlatent=None, noise_var=None, randomize_noise=True):
        if self.action[0]:
            x = self.apply_noise(x, noise_var, randomize_noise)

        x = self.bias(x)
        x = self.act(x)

        x = pixel_norm(x) if self.action[1] else x
        x = instance_norm(x) if self.action[2] else x

        if self.action[3]:
            assert dlatent is not None, "dlatent needed!"
            x = self.style_mod(x, dlatent)

        return x


class ApplyNoise(torch.nn.Module):
    """Noise input. """
    def __init__(self, num_features):
        super(ApplyNoise, self).__init__()

        self.register_parameter("scale_factor",
                                torch.nn.Parameter(torch.zeros(1, num_features, 1, 1)))

    def forward(self, feature, noise_var=None, randomize_noise=True):
        assert len(feature.shape) == 4, "not NCHW!"
        if noise_var is None or randomize_noise:
            noise_var = torch.randn(feature.shape[0], 1, *feature.shape[2:], device=feature.device)

        return feature + noise_var * self.scale_factor


class StyleMod(torch.nn.Module):
    """Style modulation. """
    def __init__(self, num_features, dlatent_size, **kwargs):
        super(StyleMod, self).__init__()

        self.affine_transform = Linear(dlatent_size, 2 * num_features, bias=True, **kwargs)

    def forward(self, feature, dlatent):
        style = self.affine_transform(dlatent)

        style = style.reshape(-1, 2, feature.size(1), 1, 1)
        feature = feature * (style[:, 0] + 1) + style[:, 1]

        return feature


class Bias(torch.nn.Module):
    """Apply bias to the given activation tensor. """
    def __init__(self, num_features, lrmul=1.):
        super(Bias, self).__init__()

        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("lrmul", torch.tensor(lrmul))

    def forward(self, x):
        bias = self.bias * self.lrmul
        if x.ndim == 2:
            return x + bias

        return x + bias.reshape(1, -1, 1, 1)
