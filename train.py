"""
Модуль для тренировки нейронной сети/модели.
"""
import tensorflow as tf
from module.project_logging import setup_logger
from module.train_model import train_model
from module.evaluate_model import evaluate_model
from module.load_train_test_val import load_train_test_val


logger = setup_logger("Learning model started")

if __name__ == '__main__':
    directory = '/mnt/d/project_practicum/dataset'
    # directory = '/mnt/d/project_practicum/mnist_images'
    batch_size = 128
    image_size = (128, 128)
    num_classes = 58
    epochs = 1000
    learning_rate = 1e-04

    train_model(
        directory=directory,
        batch_size=batch_size,
        image_size=image_size,
        num_classes = num_classes,
        epochs=epochs,
        learning_rate=learning_rate
    )
    # train_dataset, test_dataset = load_train_test_val(directory, batch_size, image_size)
    # model = tf.saved_model.load('models/model')
    # evaluate_model(model, test_dataset)