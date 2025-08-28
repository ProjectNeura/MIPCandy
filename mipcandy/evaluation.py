from dataclasses import dataclass
from typing import Callable, Sequence, Generator, override

import torch

from mipcandy.data import SupervisedDataset, MergedDataset, Loader, DatasetFromMemory
from mipcandy.inference import parse_predictant, Predictor
from mipcandy.types import SupportedPredictant


@dataclass
class EvalCase(object):
    metrics: dict[str, float]
    mask: torch.Tensor
    label: torch.Tensor
    image: torch.Tensor | None = None
    filename: str | None = None


class EvalResult(Sequence[EvalCase]):
    def __init__(self, metrics: dict[str, list[float]], masks: list[torch.Tensor], labels: list[torch.Tensor], *,
                 images: list[torch.Tensor] | None = None, filenames: list[str] | None = None) -> None:
        if len(masks) != len(labels):
            raise ValueError(f"Unmatched number of masks ({len(masks)}) and labels ({len(labels)})")
        self.metrics: dict[str, list[float]] = metrics
        self.mean_metrics: dict[str, float] = {name: sum(values) / len(values) for name, values in metrics.items()}
        self.images: list[torch.Tensor] | None = images
        self.masks: list[torch.Tensor] = masks
        self.labels: list[torch.Tensor] = labels
        self.filenames: list[str] | None = filenames

    @override
    def __len__(self) -> int:
        return len(self.masks)

    @override
    def __getitem__(self, item: int) -> EvalCase:
        return EvalCase({name: values[item] for name, values in self.metrics.items()}, self.masks[item],
                        self.labels[item], self.images[item] if self.images else None,
                        self.filenames[item] if self.filenames else None)

    def _select(self, metric: str, n: int, descending: bool) -> Generator[EvalCase, None, None]:
        o_values = self.metrics[metric]
        values = o_values.copy()
        values.sort(reverse=descending)
        for value in values[:n]:
            yield self[o_values.index(value)]

    def min(self, metric: str) -> EvalCase:
        return self.min_n(metric, 1)[0]

    def min_n(self, metric: str, n: int) -> tuple[EvalCase, ...]:
        return tuple(self._select(metric, n, False))

    def max(self, metric: str) -> EvalCase:
        return self.max_n(metric, 1)[0]

    def max_n(self, metric: str, n: int) -> tuple[EvalCase, ...]:
        return tuple(self._select(metric, n, True))


class Evaluator(object):
    def __init__(self, *metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        self._metrics: tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...] = metrics

    def _evaluate_dataset(self, x: SupervisedDataset, *, prefilled_masks: list[torch.Tensor] | None = None,
                          prefilled_labels: list[torch.Tensor] | None = None) -> EvalResult:
        metrics = {}
        masks = prefilled_masks if prefilled_masks else []
        labels = prefilled_labels if prefilled_labels else []
        for mask, label in x:
            if not prefilled_masks:
                masks.append(mask)
            if not prefilled_labels:
                labels.append(label)
            for m in self._metrics:
                if m.__name__ not in metrics:
                    metrics[m.__name__] = []
                metrics[m.__name__].append(m(mask, label).item())
        return EvalResult(metrics, masks, labels)

    def evaluate_dataset(self, x: SupervisedDataset) -> EvalResult:
        return self._evaluate_dataset(x)

    def evaluate(self, masks: SupportedPredictant, labels: SupportedPredictant) -> EvalResult:
        masks, filenames = parse_predictant(masks, Loader)
        labels, _ = parse_predictant(labels, Loader, as_label=True)
        r = self._evaluate_dataset(MergedDataset(DatasetFromMemory(masks), DatasetFromMemory(labels)),
                                   prefilled_masks=masks, prefilled_labels=labels)
        r.filenames = filenames
        return r

    def predict_and_evaluate(self, x: SupportedPredictant, labels: SupportedPredictant,
                             predictor: Predictor) -> EvalResult:
        x, filenames = parse_predictant(x, Loader)
        masks = [e.cpu() for e in predictor.predict(x)]
        r = self.evaluate(masks, labels)
        r.images = x
        r.filenames = filenames
        return r
