import numpy as np


def quotient_regression(x: np.ndarray, y: np.ndarray, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = []
    for xi, yi in zip(x, y):
        row = []
        for k in range(m + 1):
            row.append(xi ** k)
        for k in range(1, n + 1):
            row.append(-yi * xi ** k)
        matrix.append(row)
    matrix = np.array(matrix)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    return coefficients[:m + 1][::-1], np.concatenate(([1.0], coefficients[m + 1:]))[::-1]


def quotient_derivative(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    da = np.polyder(a)
    db = np.polyder(b)
    return np.polysub(np.polymul(da, b), np.polymul(a, db)), np.polymul(b, b)


def quotient_bounds(a: np.ndarray, b: np.ndarray, lower_bound: float | None, upper_bound: float | None, *,
                    x_start: float = 0, x_stop: float = 1e4, x_step: float = .01) -> tuple[float, float] | None:
    x = np.arange(x_start, x_stop, x_step)
    y = np.polyval(a, x) / np.polyval(b, x)
    mask = np.array(True, like=y)
    if lower_bound is not None:
        mask = mask & (y < lower_bound)
    if upper_bound is not None:
        mask = mask & (y > upper_bound)
    if isinstance(mask, bool):
        raise ValueError("Bounds must be specified on at least one side")
    return (float(x[mask][0]), float(x[mask][-1])) if mask.any() else None
