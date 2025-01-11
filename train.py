"""
Модуль для тренировки нейронной сети/модели.
"""
import tensorflow as tf
import numpy as np
from module.project_logging import setup_logger
from module.train_model import train_model
from module.evaluate_model import evaluate_model
from module.load_train_test import load_train_test
from module.predict_class import predict_class


logger = setup_logger("Learning model started")


def main(directory,
         batch_size,
         image_size,
         num_classes,
         epochs,
         learning_rate,
         evaluate
         ):
    train_dataset, test_dataset = load_train_test(directory, batch_size, image_size)
    if not evaluate:
        train_model(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            image_size=image_size,
            num_classes = num_classes,
            epochs=epochs,
            learning_rate=learning_rate
        )

    predicted_classes, labels = predict_class(test_dataset, batch_size, image_size)
    evaluate_model(predicted_classes, labels)


if __name__ == '__main__':
    directory = 'dataset_fixed'
    batch_size = 16
    image_size = (64, 64)   # (28, 28) (64, 64) (256, 256)
    num_classes = 128  # 58 10 для mnist
    epochs = 50
    learning_rate = 1e-03
    evaluate = False  # False - обучить и оценить, True - только оценить
    main(directory, batch_size, image_size, num_classes, epochs, learning_rate, evaluate=evaluate)
