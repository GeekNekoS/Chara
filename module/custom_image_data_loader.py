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
