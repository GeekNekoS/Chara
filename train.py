"""
Модуль для тренировки нейронной сети/модели.
"""
from module.project_logging import setup_logger
from module.load_train_test_val import load_train_test_val
from module.create_model import create_model
from module.train_model import train_model


logger = setup_logger("Learning model started")

if __name__ == '__main__':
    directory = '/mnt/d/project_practicum/dataset'
    batch_size = 32
    image_size = (150, 150)
    shuffle = True

    train_dataset, test_dataset, val_dataset = load_train_test_val(
        directory=directory,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
    )
    model = create_model(
        input_shape=image_size,
        num_classes=58
    )
