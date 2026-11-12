import numpy as np
import pytest

from slac_measurements.image.processing import (
    calc_crop_ranges,
    calc_image_centroids,
    center_images,
    compute_blob_stats,
    crop_images,
    process_images,
)

NX, NY = 100, 200
X0, Y0 = 30.0, 140.0


def make_image(x0=X0, y0=Y0, sigma_x=3.0, sigma_y=6.0):
    x, y = np.indices((NX, NY))
    return np.exp(-0.5 * (((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2))


def centroid_xy(image):
    stats = compute_blob_stats(image)
    return np.array([stats["x_center"], stats["y_center"]])


def test_calc_image_centroids_is_xy():
    images = np.stack([make_image(), make_image()])
    centroids = calc_image_centroids(images)
    assert centroids.shape == (2, 2)
    assert np.allclose(centroids, [X0, Y0], atol=1e-3)


def test_center_images_moves_beam_to_center():
    image = make_image()
    centered = center_images(image, calc_image_centroids(image))
    assert np.allclose(centroid_xy(centered), [NX // 2, NY // 2], atol=0.1)


def test_crop_ranges_and_crop_images_are_xy():
    image = make_image()
    crop_ranges = calc_crop_ranges(image, n_stds=2)
    x_range, y_range = crop_ranges
    assert x_range[0] < X0 < x_range[1]
    assert y_range[0] < Y0 < y_range[1]

    cropped = crop_images(image, crop_ranges)
    assert cropped.shape == (x_range[1] - x_range[0], y_range[1] - y_range[0])


def test_crop_images_does_not_modify_input():
    image = make_image()
    crop_ranges = np.array([[-5, 50], [100, 300]])
    original = crop_ranges.copy()
    cropped = crop_images(image, crop_ranges)
    assert cropped.shape == (50, NY - 100)
    assert np.array_equal(crop_ranges, original)


@pytest.mark.parametrize("center, crop", [(True, False), (False, True), (True, True)])
def test_offsets_recover_original_centroid(center, crop):
    image = make_image()
    processed, offsets = process_images(image, threshold=0, center=center, crop=crop)
    recovered = centroid_xy(processed) + offsets.ravel()
    assert np.allclose(recovered, [X0, Y0], atol=0.1)
