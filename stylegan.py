"""Network architectures used in StyleGAN paper. """
import torch


def _init_weight(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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

    def forward(self, latent, label=None, alpha=0.9, beta=0.7, cutoff=None, stage=None, **kwargs):
        """Args:
               alpha: probability of mixing styles during training. None = disable.
               beta: style strength multiplier for the truncation trick. None = disable.
               cutoff: number of layers for which to apply the truncation trick. None = disable.
        """
        if self.training or (beta is not None and beta == 1):
            beta = None
        if self.training or (cutoff is not None and cutoff <= 0):
            cutoff = None
        if not self.training or (alpha is not None and  alpha <= 0):
            alpha = None
        if not self.training or self.dlatent_avg_beta == 1:
            self.dlatent_avg_beta = None
        if stage is not None and stage <= 0:
            stage = self.synthesis.stage

        dlatents = self._dlatents(stage, latent, label, alpha, beta, cutoff, **kwargs)
        noises = self._noises(stage, latent.size(0), latent.device, **kwargs)

        return self.synthesis(stage, dlatents, noises, **kwargs)

    def _dlatents(self, stage, latent, label, alpha, beta, cutoff, latent2=None, section=None, **kwargs):
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
        if alpha is not None:
            _latent2 = torch.randn_like(latent)
            _dlatent2 = self.mapping(_latent2, label, **kwargs)
            mix_cutoff = torch.where(torch.rand(1) < alpha, torch.randint(num_layers, (1,)),
                                     torch.tensor([num_layers - 1])).to(device)
            dlatents = torch.where(layer_idx < mix_cutoff, dlatent, _dlatent2)

        # Multi-styles.
        if latent2 is not None and section is not None:
            dlatent2 = self.mapping(latent2, label, **kwargs)
            dlatents = torch.where(sum([layer_idx == i for i in section]), dlatent2, dlatent)

        # Apply truncation trick.
        cutoff = [cutoff, stage][cutoff is None]
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
            shape = (2, batch_size,1, resolution, resolution)
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

        self.sub_module = self._make_layers(latent_size + label_size, dlatent_size, mapping_layers, mapping_fmaps)

        self.apply(_init_weight)

    def forward(self, latent, label=None, normalize_latent=True, **_kwargs):
        if label is None:
            label = torch.empty(latent.size(0), 0, device=latent.device)

        if self.label_size:
            assert labels is not None, "labels needed!"
            label = self.label_embed(label)

        _z = torch.cat([latent, label], dim=1)
        if normalize_latent:
            _z = _z * torch.rsqrt(torch.mean(_z ** 2, dim=1, keepdim=True) + 1e-8)

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
        self._kwargs = dict(num_channels=num_channels, dlatent_size=dlatent_size, use_styles=use_styles, use_noise=use_noise)

        self.stage_0 = _InputG(self._nf(1), ilatent_size, const_input, **self._kwargs)
        self.upscales = torch.nn.ModuleDict()

        for _ in range(1, [stage, self.resolution_log2 - 1][stage is None]):
            self.__grow()

    def forward(self, stage=None, dlatents=None, noises=None, ilatent=None, **_kwargs):
        stage = [stage, self.stage][stage is None]
        if self.training and stage > self.stage:
            for _ in range(self.stage, stage):
                self.__grow()
            if next(self.stage_0.parameters()).is_cuda: self.cuda()
        assert stage <= self.stage, "stage excceeding!"

        if dlatents is None: dlatents = [[None, None]] * self.stage
        if noises is None: noises = [[None, None]] * self.stage

        _rgb = None
        x, rgb = self.stage_0(dlatents[0], noises[0], ilatent)

        for i in range(1, stage):
            _rgb = rgb
            x, rgb = self.upscales[f'stage_{i}'](x, dlatents[i], noises[i])

        return _rgb, rgb

    def __grow(self):
        """Supports progressive growing. """
        assert self.stage + 2 <= self.resolution_log2, "stage exceeding upper limit!"

        self.upscales.update({
            'stage_{}'.format(self.stage):
            UpScaleConv2d(self._nf(self.stage), self._nf(self.stage + 1), **self._kwargs),
        })
        self.stage += 1

class _InputG(torch.nn.Module):

    def __init__(self, num_features, ilatent_size=0, const_input=True, num_channels=3, **kwargs):
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

        self.torgb = torch.nn.Conv2d(num_features, num_channels, 1, 1, 0)

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

        return x, self.torgb(x)


class UpScaleConv2d(torch.nn.Module):
    """Building blocks for generator. """
    def __init__(self, fmap_in, fmap_out, num_channels=3, **kwargs):
        super(UpScaleConv2d, self).__init__()

        self.layer_0 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(fmap_in, fmap_out, 3, 1, 1, bias=False),
        )
        self.epilog_0 = EpilogueLayer(fmap_out, **kwargs)
        self.layer_1 = torch.nn.Conv2d(fmap_out, fmap_out, 3, 1, 1, bias=False)
        self.epilog_1 = EpilogueLayer(fmap_out, **kwargs)

        self.torgb = torch.nn.Conv2d(fmap_out, num_channels, 1, 1, 0)

        self.apply(_init_weight)

    def forward(self, x, dlatents=[None, None], noises=[None, None]):
        x = self.layer_0(x)
        x = self.epilog_0(x, dlatents[0], noises[0])
        x = self.layer_1(x)
        x = self.epilog_1(x, dlatents[1], noises[1])

        return x, self.torgb(x)


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
        self.instance_norm = torch.nn.InstanceNorm2d(num_features)

    def forward(self, features, dlatent):
        styles = self.affine_transform(dlatent)
        contents = self.instance_norm(features)

        styles = styles.reshape(-1, 2, contents.size(1), 1, 1)
        features = contents * (styles[:, 0] + 1) + styles[:, 1]

        return features
