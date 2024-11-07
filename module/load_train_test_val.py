import tensorflow as tf
import keras
from module.project_logging import setup_logger
from tensorflow import data as tf_data

logger = setup_logger("load_train_test_val")


def load_train_test_val(directory: str,
                        batch_size: int,
                        image_size: tuple[int, int],
                        shuffle: bool = False,
                        seed: int = 42,
                        validation_split: float = 0.2,
                        data_format: str = 'channels_last',
                        labels: str = 'inferred') -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Загружает изображения из указанной директории, автоматически разделяя их на обучающий,
    тестовый и валидационный наборы данных для последующей передачи в нейронные сети.

    Параметры:
    ----------
    directory : str
        Путь к директории, содержащей изображения, организованные по подкаталогам для каждой категории (метки).
        Имя каждой поддиректории будет интерпретировано как метка для класса изображений в этой папке.

    batch_size : int
        Размер пакета (batch) для загрузки данных. Указывает количество изображений, которые будут загружены и
        переданы сети за один шаг.

    image_size : tuple[int, int]
        Размер загружаемых изображений (в пикселях) после масштабирования. Формат: (высота, ширина).
        Все изображения будут масштабированы до этого размера, чтобы обеспечить единообразие данных.

    shuffle : bool, optional (по умолчанию False)
        Параметр для перемешивания изображений. Если True, изображения будут перемешаны перед загрузкой,
        что полезно для предотвращения возможных последовательных зависимостей.

    seed : int, optional (по умолчанию 42)
        Значение для генератора случайных чисел, чтобы обеспечить воспроизводимость перемешивания
        и разделения набора данных при указании `shuffle=True` или `validation_split`.

    validation_split : float, optional (по умолчанию 0.2)
        Доля данных, отводимая для валидационного набора. Должно быть числом от 0 до 1, где 0 означает отсутствие
        валидационного набора, а 1 — что все данные будут использованы только для валидации.

    data_format : str, optional (по умолчанию 'channels_last')
        Формат данных, определяющий, где располагается канал цвета в загружаемых изображениях.
        `'channels_last'` — канал цвета располагается в последней оси (формат [height, width, channels]).

    labels : str, optional (по умолчанию 'inferred')
        Параметр, определяющий способ присвоения меток (labels) изображениям.
        `'inferred'` — метки определяются автоматически на основе структуры директорий.
        Если задать значение `None`, метки не будут присваиваться.

    Возвращаемое значение:
    ----------------------
    tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        Кортеж, содержащий три объекта `tf.data.Dataset` для обучающего (`train_dataset`), тестового (`test_dataset`)
        и валидационного (`val_dataset`) наборов данных, которые можно использовать для тренировки, проверки и
        оценки модели.

    Логика работы:
    --------------
    - Сначала функция загружает данные из указанной директории, выполняя разделение на обучающий и валидационный
      наборы данных, используя метод `image_dataset_from_directory`.
    - Затем обучающий набор данных дополнительно делится на обучающий и тестовый наборы в соотношении 75% на 25%
      с использованием `split_dataset`, чтобы выделить часть данных для тестирования.
    - В конце функция возвращает три разделённых набора данных для дальнейшего использования.
    """
    logger.info('Loading data')
    train_dataset, test_dataset = keras.utils.image_dataset_from_directory(directory,
                                                                          batch_size=batch_size,
                                                                          image_size=image_size,
                                                                          shuffle=shuffle,
                                                                          seed=seed,
                                                                          validation_split=validation_split,
                                                                          data_format=data_format,
                                                                          labels=labels,
                                                                          subset='both')
    train_dataset, val_dataset = keras.utils.split_dataset(train_dataset,
                                                            left_size=0.75,
                                                            shuffle=shuffle,
                                                            seed=seed)
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf_data.AUTOTUNE)
    logger.info('Data loaded')
    return train_dataset, val_dataset, test_dataset
