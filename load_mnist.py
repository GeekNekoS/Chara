import os
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

# Загрузка набора данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Указание пути, где будут сохранены изображения
save_dir = '/mnt/d/project_practicum/mnist_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Создаем папки для каждой категории (цифры от 0 до 9)
for i in range(10):
    category_dir = os.path.join(save_dir, str(i))
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

# Сохраняем изображения из обучающего набора в соответствующие папки
for i in range(len(X_train)):
    img = Image.fromarray(X_train[i])  # Преобразуем массив в изображение
    label = y_train[i]  # Метка (цифра)
    img_path = os.path.join(save_dir, str(label), f'{i}.png')
    img.save(img_path)

print(f"Изображения сохранены в папку {save_dir}")
