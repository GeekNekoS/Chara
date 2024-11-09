"""
Модуль, изменяющий формат модели с .keras на savedmodel tensorflow
"""
import tensorflow as tf


if __name__ == '__main__':
    path = '/mnt/c/Users/unive/PycharmProjects/Nastya_labs/Chara/models/'

    # Загрузка модели Keras из файла (форматы .keras или .h5)
    keras_model = tf.keras.models.load_model(path + 'model.keras')

    # Сохранение модели в формате SavedModel
    tf.saved_model.save(keras_model, path + 'model')
