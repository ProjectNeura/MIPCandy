import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TrainingAugmentation(nn.Module):
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        image, label = x['image'], x['label']
        spatial_dims = image.ndim - 1
        for d in range(spatial_dims):
            if random.random() < 0.5:
                image = image.flip(d + 1)
                label = label.flip(d)

        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, [-2, -1])
            label = torch.rot90(label, k, [-2, -1])

        if random.random() < 0.1:
            noise_std = random.uniform(0, 0.1)
            image = image + torch.randn_like(image) * noise_std

        if random.random() < 0.15:
            factor = random.uniform(0.75, 1.25)
            image = image * factor

        if random.random() < 0.15:
            mean = image.mean()
            factor = random.uniform(0.75, 1.25)
            image = (image - mean) * factor + mean

        if random.random() < 0.15:
            gamma = random.uniform(0.7, 1.5)
            min_val = image.min()
            rng = image.max() - min_val + 1e-7
            image = ((image - min_val) / rng).pow(gamma) * rng + min_val

        if random.random() < 0.25:
            zoom = random.uniform(0.5, 1.0)
            orig_shape = image.shape[1:]
            small = [max(1, int(s * zoom)) for s in orig_shape]
            mode = 'trilinear' if len(orig_shape) == 3 else 'bilinear'
            image = F.interpolate(
                F.interpolate(image.unsqueeze(0), size=small, mode=mode, align_corners=False),
                size=orig_shape, mode=mode, align_corners=False
            ).squeeze(0)

        return {"image": image, "label": label}
