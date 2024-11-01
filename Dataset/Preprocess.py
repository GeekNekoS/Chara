import tensorflow as tf
import keras


def load_and_split_dataset(directory: str,
                           batch_size: int,
                           image_size: tuple[int, int],
                           shuffle: bool = False,
                           seed: int = 42,
                           validation_split: float = 0.2,
                           data_format: str = 'channels_last',
                           labels: str = 'inferred') -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
                                                            left_size=0.8,
                                                            shuffle=shuffle,
                                                            seed=seed)
    return train_dataset, test_dataset, val_dataset
