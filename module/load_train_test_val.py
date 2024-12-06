import tensorflow as tf
import keras
from module.project_logging import setup_logger

logger = setup_logger("load_train_test_val")


def load_train_test_val(directory: str,
                        batch_size: int,
                        image_size: tuple[int, int],
                        shuffle: bool = True,
                        seed: int = 42,
                        validation_split: float = 0.2,
                        data_format: str = 'channels_last',
                        labels: str = 'inferred') -> tuple[tf.data.Dataset, tf.data.Dataset]:
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
                                                                          # batch_size=batch_size,
                                                                          image_size=image_size,
                                                                          shuffle=shuffle,
                                                                          seed=seed,
                                                                          validation_split=validation_split,
                                                                          data_format=data_format,
                                                                          labels=labels,
                                                                          label_mode='categorical',
                                                                          # class_names=["0", "1"],  # список классов, которые необходимо загрузить
                                                                          subset='both',
                                                                          verbose=True
                                                                          )

    #
    # # Prefetching samples in GPU memory helps maximize GPU utilization.
    # # train_dataset = train_dataset.prefetch(tf_data.AUTOTUNE)
    # # test_dataset = test_dataset.prefetch(tf_data.AUTOTUNE)
    # # # OneHotEncoding целевых меток
    # # train_dataset = train_dataset.unbatch()
    # train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=10)))
    # # train_dataset = train_dataset.batch(batch_size)
    # # # val_dataset = val_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=58)))
    # # test_dataset = test_dataset.unbatch()
    # test_dataset = test_dataset.map(lambda x, y: (x, tf.one_hot(y, depth=10)))
    # # test_dataset = test_dataset.batch(batch_size)


    # data_loader = CustomImageDataLoader(data_dir=directory, img_size=image_size, batch_size=batch_size, test_split=validation_split)
    # train_dataset = data_loader.get_train_data()
    # test_dataset = data_loader.get_test_data()
    # class_names = data_loader.get_class_names()
    # print("Class names:", class_names)

    logger.info('Data loaded')
    return train_dataset, test_dataset


import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.utils import to_categorical
from concurrent.futures import ProcessPoolExecutor, as_completed


class CustomImageDataLoader:
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32, test_split=0.2, num_workers=16):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_workers = num_workers  # Количество потоков для загрузки
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.num_classes = len(self.class_names)
        self._prepare_data()

    def _load_single_image(self, file_path, label):
        # Загружаем и обрабатываем одно изображение
        image = load_img(file_path, target_size=self.img_size)
        image = img_to_array(image) / 255.0  # Нормализация
        return image, label

    def _load_images_from_folder(self, folder_path, label):
        images, labels = [], []
        # Загрузка всех изображений из текущей папки
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                image, lbl = self._load_single_image(file_path, label)
                images.append(image)
                labels.append(lbl)
        return images, labels

    def _load_images_and_labels(self):
        images, labels = [], []

        # Список путей к папкам классов
        folder_paths_labels = [
            (os.path.join(self.data_dir, class_name), label)
            for label, class_name in enumerate(self.class_names)
        ]

        # Используем ProcessPoolExecutor для параллельной загрузки изображений из папок
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._load_images_from_folder, folder_path, label)
                for folder_path, label in folder_paths_labels
            ]

            # Собираем результаты и выводим прогресс по папкам
            processed_folders = 0
            total_folders = len(folder_paths_labels)

            for future in as_completed(futures):
                folder_images, folder_labels = future.result()
                images.extend(folder_images)
                labels.extend(folder_labels)

                # Обновляем прогресс после обработки каждой папки
                processed_folders += 1
                print(f"Папок обработано: {processed_folders}/{total_folders}")
        return np.array(images), np.array(labels)

    def _prepare_data(self):
        images, labels = self._load_images_and_labels()

        # Применяем one-hot encoding к меткам
        labels = to_categorical(labels, num_classes=self.num_classes)

        # Разделяем данные на обучающую и тестовую выборки
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=self.test_split, stratify=labels)

        # Преобразуем в tf.data.Dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(self.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)

    def get_train_data(self):
        return self.train_dataset

    def get_test_data(self):
        return self.test_dataset

    def get_class_names(self):
        return self.class_names

