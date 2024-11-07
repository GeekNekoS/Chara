import tensorflow as tf
import keras
from module.project_logging import setup_logger

logger = setup_logger("load_train_test_val")


def load_train_test_val(directory: str,
                        batch_size: int,
                        image_size: tuple[int, int],
                        shuffle: bool = False,
                        seed: int = 42,
                        validation_split: float = 0.2,
                        data_format: str = 'channels_last',
                        labels: str = 'inferred') -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    logger.info('Loading data')
    train_dataset, val_dataset = keras.utils.image_dataset_from_directory(directory,
                                                                          batch_size=batch_size,
                                                                          image_size=image_size,
                                                                          shuffle=shuffle,
                                                                          seed=seed,
                                                                          validation_split=validation_split,
                                                                          data_format=data_format,
                                                                          labels=labels,
                                                                          subset='both')
    train_dataset, test_dataset = keras.utils.split_dataset(train_dataset,
                                                            left_size=0.75,
                                                            shuffle=shuffle,
                                                            seed=seed)
    logger.info('Data loaded')
    return train_dataset, test_dataset, val_dataset
