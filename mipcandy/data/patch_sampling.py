import numpy as np
import torch


def compute_foreground_locations(label: torch.Tensor, *, ignore_label: int = 0) -> dict[int, torch.Tensor]:
    foreground_locs = {}
    unique_classes = torch.unique(label)
    for class_id in unique_classes:
        if class_id == ignore_label:
            continue
        coords = torch.nonzero(label == class_id, as_tuple=False)
        if len(coords) > 0:
            foreground_locs[int(class_id)] = coords
    return foreground_locs


def sample_random_bbox(shape: tuple[int, ...], patch_size: tuple[int, ...],
                       rng: np.random.RandomState) -> tuple[int, ...]:
    return tuple(
        int(rng.randint(-patch_size[d] // 4, shape[d] + patch_size[d] // 4 - patch_size[d]))
        for d in range(len(shape))
    )


def crop_and_pad(image: torch.Tensor, label: torch.Tensor, bbox_start: tuple[int, ...],
                 patch_size: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = len(patch_size)
    num_prefix_dims = image.ndim - ndim
    slices = [slice(None)] * num_prefix_dims
    pads = []

    for d in range(ndim):
        start = bbox_start[d]
        end = start + patch_size[d]
        img_size = image.shape[num_prefix_dims + d]

        valid_start = max(0, start)
        valid_end = min(img_size, end)

        pad_before = max(0, -start)
        pad_after = max(0, end - img_size)

        slices.append(slice(valid_start, valid_end))
        pads.extend([pad_before, pad_after])

    img_crop = image[tuple(slices)]
    lbl_crop = label[tuple(slices)]

    if any(p > 0 for p in pads):
        pad_tuple = tuple(reversed(pads))
        img_crop = torch.nn.functional.pad(img_crop, pad_tuple, value=0)
        lbl_crop = torch.nn.functional.pad(lbl_crop, pad_tuple, value=0)

    return img_crop, lbl_crop


def sample_random_patch(image: torch.Tensor, label: torch.Tensor, properties: dict, patch_size: tuple[int, ...],
                        force_foreground: bool, rng: np.random.RandomState) -> tuple[torch.Tensor, torch.Tensor]:
    ndim = len(patch_size)
    shape = image.shape[1:]

    if force_foreground and 'foreground_locations' in properties:
        foreground_locs = properties['foreground_locations']
        if foreground_locs:
            class_id = rng.choice(list(foreground_locs.keys()))
            coords = foreground_locs[class_id]

            if len(coords) > 0:
                center_idx = int(rng.randint(0, len(coords)))
                center = coords[center_idx]

                bbox_start = tuple(
                    max(0, int(center[d]) - patch_size[d] // 2)
                    for d in range(ndim)
                )
            else:
                bbox_start = sample_random_bbox(shape, patch_size, rng)
        else:
            bbox_start = sample_random_bbox(shape, patch_size, rng)
    else:
        bbox_start = sample_random_bbox(shape, patch_size, rng)

    return crop_and_pad(image, label, bbox_start, patch_size)
