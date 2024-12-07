"""
Модуль для тренировки нейронной сети/модели.
"""
import tensorflow as tf
import numpy as np
from module.project_logging import setup_logger
from module.train_model import train_model
from module.evaluate_model import evaluate_model
from module.load_train_test_val import load_train_test_val
from module.predict_class import predict_class


logger = setup_logger("Learning model started")


def main(directory,
         batch_size,
         image_size,
         num_classes,
         epochs,
         learning_rate
         ):
    train_dataset, test_dataset = load_train_test_val(directory, batch_size, image_size)
    # train_model(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     batch_size=batch_size,
    #     image_size=image_size,
    #     num_classes = num_classes,
    #     epochs=epochs,
    #     learning_rate=learning_rate
    # )

    predicted_classes, labels = predict_class(test_dataset, batch_size, image_size)
    evaluate_model(predicted_classes, labels)

if __name__ == '__main__':
    # directory = '/mnt/d/project_practicum/dataset'
    # directory = '/mnt/d/project_practicum/dataset_augmented'
    directory = '/mnt/d/project_practicum/mnist_images'
    batch_size = 256
    image_size = (28, 28)   # (28, 28) (128, 128)
    num_classes = 10
    epochs = 100
    learning_rate = 1e-03

    main(directory, batch_size, image_size, num_classes, epochs, learning_rate)
