from typing import override

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader as BGDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from mipcandy.data.dataset import RandomPatchDataset
from mipcandy.data.patch_sampling import sample_random_patch


class RandomPatchDataLoader(BGDataLoader):

    def __init__(self, dataset: RandomPatchDataset, batch_size: int, patch_size: tuple[int, ...], *,
                 oversample_foreground_percent: float = 0.33, num_iterations_per_epoch: int = 250,
                 probabilistic_oversampling: bool = False, num_workers: int = 0, seed: int | None = None) -> None:
        super().__init__(list(range(len(dataset))), batch_size, seed_for_shuffle=seed, shuffle=False, infinite=False)

        self.dataset: RandomPatchDataset = dataset
        self.patch_size: tuple[int, ...] = patch_size
        self.num_iterations: int = num_iterations_per_epoch

        self._oversample_fg: float = oversample_foreground_percent
        self._probabilistic: bool = probabilistic_oversampling
        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self._num_workers: int = num_workers
        self._seed: int | None = seed
        self._augmenter: MultiThreadedAugmenter | None = None

        if num_workers > 0:
            import sys
            if sys.platform == 'win32':
                from rich.console import Console
                console = Console()
                console.print(f"[yellow]Warning: num_workers={num_workers} is not supported on Windows due to pickle limitations. Setting num_workers=0.[/yellow]")
                self._num_workers = 0
            else:
                seeds = [seed + i if seed is not None else None for i in range(num_workers)]
                self._augmenter = MultiThreadedAugmenter(
                    self, None, num_processes=num_workers, num_cached_per_queue=2, seeds=seeds, pin_memory=False
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_augmenter'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._num_workers > 0:
            seeds = [self._seed + i if self._seed is not None else None for i in range(self._num_workers)]
            self._augmenter = MultiThreadedAugmenter(
                self, None, num_processes=self._num_workers, num_cached_per_queue=2, seeds=seeds, pin_memory=False
            )

    def _get_do_oversample(self, sample_idx: int) -> bool:
        if self._probabilistic:
            return self._rng.uniform() < self._oversample_fg
        else:
            return sample_idx >= round(self.batch_size * (1 - self._oversample_fg))

    @override
    def generate_train_batch(self) -> dict[str, np.ndarray]:
        indices = self._rng.choice(len(self.dataset), self.batch_size, replace=True)

        images = []
        labels = []

        for j, idx in enumerate(indices):
            img, lbl, props = self.dataset.load_with_properties(int(idx))
            force_fg = self._get_do_oversample(j)

            patch_img, patch_lbl = sample_random_patch(img, lbl, props, self.patch_size, force_fg, self._rng)

            images.append(patch_img.cpu().numpy())
            labels.append(patch_lbl.cpu().numpy())

        return {'data': np.stack(images), 'target': np.stack(labels)}

    @override
    def __iter__(self):
        if self._augmenter is not None:
            self._augmenter.restart()
            count = 0
            for batch_dict in self._augmenter:
                if count >= len(self):
                    break
                yield torch.from_numpy(batch_dict['data']), torch.from_numpy(batch_dict['target'])
                count += 1
        else:
            for i in range(len(self)):
                batch = self.generate_train_batch()
                yield torch.from_numpy(batch['data']), torch.from_numpy(batch['target'])

    @override
    def __len__(self) -> int:
        return self.num_iterations
