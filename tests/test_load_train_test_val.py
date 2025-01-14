from module.load_train_test import load_train_test
import pytest
import tensorflow as tf
import os


BATCH_SIZE = 8
IMAGES_NUMBER = 40
IMAGE_SIZE = (140, 140)
CHANNELS_NUMBER = 3


@pytest.fixture
def create_dummy_data(tmp_path):
    """
    Create directories with random images.
    """
    os.makedirs(tmp_path / "class_a", exist_ok=True)
    os.makedirs(tmp_path / "class_b", exist_ok=True)
    for i in range(IMAGES_NUMBER):
        image = tf.random.uniform((*IMAGE_SIZE, CHANNELS_NUMBER))
        tf.keras.preprocessing.image.save_img(str(tmp_path / "class_a" / f"{i}.jpg"), image)
        tf.keras.preprocessing.image.save_img(str(tmp_path / "class_b" / f"{i + IMAGES_NUMBER}.jpg"), image)
    return str(tmp_path)


def test_output_shapes(create_dummy_data):
    """
    Checks that the function returns three datasets.
    """
    train_ds, test_ds, val_ds = load_train_test(create_dummy_data,
                                                BATCH_SIZE,
                                                IMAGE_SIZE)
    assert isinstance(train_ds, tf.data.Dataset), "Train dataset must be a tf.data.Dataset object"
    assert isinstance(test_ds, tf.data.Dataset), "Test dataset must be a tf.data.Dataset object"
    assert isinstance(val_ds, tf.data.Dataset), "Validation dataset must be a tf.data.Dataset object"


def test_split_proportions(create_dummy_data):
    """
    Checks that the data split proportions are correct.
    """
    train_ds, test_ds, val_ds = load_train_test(create_dummy_data,
                                                BATCH_SIZE,
                                                IMAGE_SIZE)
    train_size = sum(1 for _ in train_ds)
    test_size = sum(1 for _ in test_ds)
    val_size = sum(1 for _ in val_ds)
    assert train_size / (train_size + test_size) == pytest.approx(0.75, rel=0.1), 'train/test split proportions must be 75%/25%'
    assert val_size / (train_size + test_size + val_size) == pytest.approx(0.2, rel=0.1), '(train+test)/val split proportions must be 80%/20%'


def test_batch_size(create_dummy_data):
    """
    Checks that the data packet size matches the given batch size.
    """
    train_ds, test_ds, val_ds = load_train_test(create_dummy_data,
                                                BATCH_SIZE,
                                                IMAGE_SIZE)
    for images, labels in train_ds.take(1):
        assert images.shape[0] == BATCH_SIZE, f'The packet size in the training set should be {BATCH_SIZE}'


def test_image_size(create_dummy_data):
    """
    Checks that images are loaded at the correct size.
    """
    train_ds, test_ds, val_ds = load_train_test(create_dummy_data,
                                                BATCH_SIZE,
                                                IMAGE_SIZE)
    for images, labels in train_ds.take(1):
        assert images.shape[1:3] == IMAGE_SIZE, f'Image size should be {IMAGE_SIZE}'


def test_channel_format(create_dummy_data):
    """
    Checks that images are loaded with the correct channel format.
    """
    train_ds, test_ds, val_ds = load_train_test(create_dummy_data,
                                                BATCH_SIZE,
                                                IMAGE_SIZE)
    for images, labels in train_ds.take(1):
        assert images.shape[-1] == CHANNELS_NUMBER, f"Images must have {CHANNELS_NUMBER} channels"
