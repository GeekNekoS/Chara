"""
Модуль для тренировки нейронной сети/модели.
"""
from module.project_logging import setup_logger
from module.train_model import train_model


logger = setup_logger("Learning model started")

if __name__ == '__main__':
    directory = '/mnt/d/project_practicum/dataset'
    batch_size = 32
    image_size = (64, 64)
    num_classes = 58
    epochs = 100

    train_model(
        directory=directory,
        batch_size=batch_size,
        image_size=image_size,
        num_classes = num_classes,
        epochs=epochs
    )