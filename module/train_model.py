from project_logging import logger
from load_train_test_val import load_train_test_val
from create_model import create_model
from evaluate_model import evaluate_model
import numpy as np
import keras
import os


def train_model(directory: str,
                batch_size: int,
                image_size: tuple[int, int]):
    logger.info(f"train_model: starts with directory: {directory}, batch_size: {batch_size}, image_size: {image_size}")
    try:
        train_dataset, test_dataset, val_dataset = load_train_test_val(directory, batch_size, image_size)
        model = create_model(np.shape(list(train_dataset.take(1))[0][0]), sum([1 for _ in os.listdir(directory)]))
        model.compile()
        model.fit(train_dataset,
                  test_dataset,
                  batch_size,
                  callbacks=keras.callbacks.EarlyStopping())
        evaluate_model(model, val_dataset)
        logger.info("train_model: training completed successfully")
    except Exception as exc:
        logger.error(f"An error occurred during training: {exc}")
