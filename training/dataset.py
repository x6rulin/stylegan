"""Multi-resolution input data pipline. """
import os

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class _MultiResTransform:
    """Configurable resolution transforms. """
    def __init__(self, num_channels: int = 3, resolution: [int, tuple] = None):
        self.num_channels = num_channels
        self.resolution = resolution

    def _transform(self, image):
        _trans_list = []
        if self.resolution is not None:
            _trans_list.extend([
                transforms.Resize(self.resolution),
                transforms.CenterCrop(self.resolution),
            ])
        _trans_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * self.num_channels, (0.5,) * self.num_channels, True),
        ])

        transform = transforms.Compose(_trans_list)
        return transform(image)


class FFHQDataset(Dataset, _MultiResTransform):
    """Flickr-Faces-HQ dataset. """
    def __init__(self, image_root: str, num_channels: int = 3, resolution: [int, tuple] = None):
        super(FFHQDataset, self).__init__(num_channels, resolution)

        self.data = []
        for root, _, files in os.walk(image_root):
            if root != image_root:
                self.data.extend([os.path.join(root, name) for name in files])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return self._transform(Image.open(self.data[item]))
