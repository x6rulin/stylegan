"""Network architectures used in StyleGAN paper. """
import torch


def _init_weight(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class GMapping(torch.nn.Module):
    """Mapping network used in StyleGAN. """
    def __init__(self, latent_size=512, label_size=0, dlatent_size=512, mapping_layers=8, mapping_fmaps=512):
        super(GMapping, self).__init__()

        self.label_size = label_size

        if label_size:
            self.label_embed = torch.nn.Linear(label_size, latent_size, bias=False)

        self.sub_module = self._make_layers(latent_size + label_size, dlatent_size, mapping_layers, mapping_fmaps)

    def forward(self, latents, labels=None, normalize_latent=True):
        if labels is None:
            labels = torch.empty(latents.size(0), 0, device=latents.device)

        if self.label_size:
            assert labels is not None, "labels needed!"
            labels = self.label_embed(labels)

        _z = torch.cat([latents, labels], dim=1)
        if normalize_latent:
            _z = _z * torch.rsqrt(torch.mean(_z ** 2, dim=1, keepdim=True) + 1e-8)

        _w = self.sub_module(_z)
        return _w

    @staticmethod
    def _make_layers(in_features, out_features, mapping_layers, mapping_fmaps):
        layers = []

        _mid_features = [in_features] + [mapping_fmaps] * (mapping_layers - 1) + [out_features]
        for _in, _out in zip(_mid_features[:-1], _mid_features[1:]):
            layers.extend([torch.nn.Linear(_in, _out), torch.nn.LeakyReLU(0.2, inplace=True)])

        return torch.nn.Sequential(*layers)


class StyleMod(torch.nn.Module):
    """Style modulation, which controls the generator through AdaIN at corresponding convolution layer. """
    def __init__(self, dlatent_size, num_features):
        super(StyleMod, self).__init__()

        self.affine_transform = torch.nn.Linear(dlatent_size, 2 * num_features)
        self.instance_norm = torch.nn.InstanceNorm2d(num_features)

    def forward(self, features, dlatents):
        styles = self.affine_transform(dlatents)
        contents = self.instance_norm(features)

        styles = styles.reshape(-1, 2, contents.size(1), 1, 1)
        features = contents * (styles[:, 0] + 1) + styles[:, 1]

        return features


class ApplyNoise(torch.nn.Module):
    """Noise input for corresponding convolution layer. """
    def __init__(self, num_features):
        super(ApplyNoise, self).__init__()

        self.scaling_factors = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, features, noise_var=None, random_noise=True):
        if random_noise:
            noise = torch.randn(features.shape[0], 1, *features.shape[2:])
        else:
            assert noise_var is not None, "noise needed!"
            noise = noise_var

        return features + noise * self.scaling_factors


class UpsampleLayer(torch.nn.Module):
    """Bilinear upsampling layer. """
    def __init__(self, scale_factor=2):
        super(UpsampleLayer, self).__init__()

        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        return self.upsample(x)


class GSynthesis(torch.nn.Module):
    """Synthesis network used in StyleGAN. """
    def __init__(self, num_channels=3, resolution=1024, fmap_base=8192, fmap_decay=1.0, fmap_max=512, \
                 const_input=True, dlatent_size=512, use_styles=True, use_noise=True, randomize_noise=True):
        super(GSynthesis, self).__init__()

        pass

    def forward(self, x):
        pass

    def _set_states(self, const_input, use_styles, use_noise, randomize_noise):
        self.const_input = const_input
        self.use_styles = use_styles
        self.use_noise = use_noise
        self.randomize_noise = randomize_noise
