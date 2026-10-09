"""Tests for the MNIST with NumPy exercises.

They run the exercise notebook, mnist_numpy.ipynb, and check the names it defines.
Set MNIST_NOTEBOOK to test another notebook, for example the solution:

    MNIST_NOTEBOOK=mnist_numpy_solution.ipynb uv run pytest
"""

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
from IPython.core.interactiveshell import InteractiveShell

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (REPO_ROOT / os.environ.get("MNIST_NOTEBOOK", "mnist_numpy.ipynb")).resolve()


@pytest.fixture(scope="module")
def run() -> dict[str, Any]:
    """Run every code cell once, in order, and keep going after a failing cell."""
    plt.show = lambda *args, **kwargs: plt.close("all")
    shell = InteractiveShell.instance()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    failed_cells = []
    previous_directory = Path.cwd()
    os.chdir(NOTEBOOK.parent)
    try:
        for number, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            result = shell.run_cell(cell.source, silent=True)
            if not result.success:
                error = result.error_in_exec or result.error_before_exec
                failed_cells.append(f"cell {number}: {type(error).__name__}: {error}")
    finally:
        os.chdir(previous_directory)
    return {"namespace": shell.user_ns, "failed_cells": failed_cells}


@pytest.fixture(scope="module")
def ns(run: dict[str, Any]) -> dict[str, Any]:
    return run["namespace"]


def get(ns: dict[str, Any], name: str) -> Any:
    if name not in ns:
        pytest.fail(f"`{name}` is not defined yet: solve the exercise that asks for it")
    return ns[name]


def test_notebook_runs_without_errors(run: dict[str, Any]) -> None:
    assert not run["failed_cells"], "Some cells raise an error:\n" + "\n".join(run["failed_cells"])


def test_matrix_first_100_labels(ns: dict[str, Any]) -> None:
    labels = get(ns, "matrix_first_100_labels")
    assert labels.shape == (10, 10)
    assert np.array_equal(labels.ravel(), ns["y_train"][:100])


def test_flattening(ns: dict[str, Any]) -> None:
    X_train_flat = get(ns, "X_train_flat")
    X_test_flat = get(ns, "X_test_flat")
    assert X_train_flat.shape == (60_000, 784)
    assert X_test_flat.shape == (10_000, 784)
    assert np.array_equal(X_train_flat[0], ns["X_train"][0].ravel())


def test_plot_image_accepts_flat_and_square_images(ns: dict[str, Any]) -> None:
    plot_image = get(ns, "plot_image")
    plot_image(ns["X_train"][0])
    plot_image(ns["X_train"][0].ravel())


def test_n_touching_border(ns: dict[str, Any]) -> None:
    X_train = ns["X_train"]
    expected = (
        X_train[:, 0, :].any(axis=1)
        | X_train[:, -1, :].any(axis=1)
        | X_train[:, :, 0].any(axis=1)
        | X_train[:, :, -1].any(axis=1)
    ).sum()
    assert get(ns, "n_touching_border") == expected


def test_n_overflowed_pixels(ns: dict[str, Any]) -> None:
    assert get(ns, "n_overflowed_pixels") == (ns["X_train"][0] >= 128).sum()


@pytest.mark.parametrize("split", ["train", "test"])
def test_onehot_encoding(ns: dict[str, Any], split: str) -> None:
    onehot = get(ns, f"y_{split}_onehot")
    labels = ns[f"y_{split}"]
    assert onehot.shape == (len(labels), 10)
    assert set(np.unique(onehot)) <= {0, 1}
    assert np.all(onehot.sum(axis=1) == 1)
    assert np.array_equal(onehot.argmax(axis=1), labels)


def test_normalization(ns: dict[str, Any]) -> None:
    X_train_norm = get(ns, "X_train_norm")
    assert X_train_norm.shape == (60_000, 28, 28)
    assert X_train_norm.dtype == np.float32
    assert X_train_norm.min() == 0
    assert X_train_norm.max() == 1


def test_normalized_statistics(ns: dict[str, Any]) -> None:
    assert np.isclose(get(ns, "pixel_norm_mean"), 0.130660, atol=1e-5)
    assert np.isclose(get(ns, "pixel_norm_std"), 0.308108, atol=1e-5)


def test_standardization_uses_training_statistics(ns: dict[str, Any]) -> None:
    X_train, X_test = ns["X_train"], ns["X_test"]
    X_train_standardized = get(ns, "X_train_standardized")
    X_test_standardized = get(ns, "X_test_standardized")
    assert np.isclose(X_train_standardized.mean(), 0, atol=1e-6)
    assert np.isclose(X_train_standardized.std(), 1, atol=1e-6)
    expected_test = (X_test - X_train.mean()) / X_train.std()
    assert np.allclose(X_test_standardized, expected_test, atol=1e-5), (
        "Standardise the test images with the mean and standard deviation of the training images"
    )


def test_standardizing_normalized_images_gives_the_same_result(ns: dict[str, Any]) -> None:
    assert np.allclose(get(ns, "X_train_norm_standardized"), get(ns, "X_train_standardized"), atol=1e-5)


def test_normalize_with_loops(ns: dict[str, Any]) -> None:
    normalize_with_loops = get(ns, "normalize_with_loops")
    sample = ns["X_train"][:3]
    assert np.allclose(normalize_with_loops(sample), sample / 255)
    assert get(ns, "speedup") > 5


def test_noisy_images(ns: dict[str, Any]) -> None:
    X_train = ns["X_train"]
    X_train_noisy = get(ns, "X_train_noisy")
    assert X_train_noisy.shape == X_train.shape
    assert X_train_noisy.dtype == np.uint8
    assert not np.array_equal(X_train_noisy, X_train)
    assert get(ns, "add_gaussian_noise")(X_train[:2], std=50).dtype == np.uint8


def test_binarize_image(ns: dict[str, Any]) -> None:
    binarize_image = get(ns, "binarize_image")
    image = ns["X_train"][0]
    binary = binarize_image(image, threshold=128)
    assert binary.shape == (28, 28)
    assert np.array_equal(binary, (image > 128).astype(np.uint8))
    assert binarize_image(ns["X_train"][:5], threshold=128).shape == (5, 28, 28)


@pytest.mark.parametrize("name", ["X_train_binarized", "X_train_noisy_binarized"])
def test_binarized_sets(ns: dict[str, Any], name: str) -> None:
    binarized = get(ns, name)
    assert binarized.shape == (60_000, 28, 28)
    assert set(np.unique(binarized)) == {0, 1}


@pytest.mark.parametrize(("dx", "dy"), [(10, 0), (-10, 0), (0, 10), (0, -10), (3, -2), (0, 0)])
def test_shift_image(ns: dict[str, Any], dx: int, dy: int) -> None:
    shift_image = get(ns, "shift_image")
    image = ns["X_train"][0]
    expected = np.roll(image, (dy, dx), axis=(0, 1))
    if dy > 0:
        expected[:dy, :] = 0
    elif dy < 0:
        expected[dy:, :] = 0
    if dx > 0:
        expected[:, :dx] = 0
    elif dx < 0:
        expected[:, dx:] = 0
    assert np.array_equal(shift_image(image, dx=dx, dy=dy), expected)


def test_shifted_right_10(ns: dict[str, Any]) -> None:
    X_train = ns["X_train"]
    shifted = get(ns, "X_train_shifted_right_10")
    assert shifted.shape == X_train.shape
    assert not shifted[:, :, :10].any()
    assert np.array_equal(shifted[:, :, 10:], X_train[:, :, :-10])


def test_find_limits(ns: dict[str, Any]) -> None:
    find_limits = get(ns, "find_limits")
    X_train = ns["X_train"]
    assert tuple(find_limits(X_train[0])) == (5, 24, 4, 23)
    for image in X_train[1:20]:
        rows, cols = np.nonzero(image)
        assert tuple(find_limits(image)) == (rows.min(), rows.max(), cols.min(), cols.max())


def test_crop_image_limits(ns: dict[str, Any]) -> None:
    crop_image_limits = get(ns, "crop_image_limits")
    X_train = ns["X_train"]
    assert crop_image_limits(X_train[0]).shape == (20, 20)
    for image in X_train[:20]:
        assert crop_image_limits(image).sum() == image.sum(), "The crop must keep all the ink"


def test_random_shift_keeps_every_digit_in_the_frame(ns: dict[str, Any]) -> None:
    X_train = ns["X_train"]
    shifted = get(ns, "X_train_shifted")
    assert shifted.shape == X_train.shape
    assert np.array_equal(shifted.sum(axis=(1, 2)), X_train.sum(axis=(1, 2))), "Some ink left the frame"
    moved = np.any(shifted != X_train, axis=(1, 2)).mean()
    assert moved > 0.8, "Most images should actually move"


def test_augmented_set(ns: dict[str, Any]) -> None:
    X_train, y_train = ns["X_train"], ns["y_train"]
    augmented = get(ns, "X_train_augmented")
    labels = get(ns, "y_train_augmented")
    assert augmented.shape == (300_000, 28, 28)
    assert np.array_equal(augmented[:60_000], X_train)
    assert set(np.unique(augmented[120_000:180_000])) == {0, 255}, "Scale the binarised images to 0 and 255"
    assert np.array_equal(labels, np.tile(y_train, 5))
