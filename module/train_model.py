from load_train_test_val import load_train_test_val
from create_model import create_model
from evaluate_model import evaluate_model
import numpy as np
import keras


def train_model(directory: str,
                batch_size: int,
                image_size: tuple[int, int]):
    train_dataset, test_dataset, val_dataset = load_train_test_val(directory, batch_size, image_size)
    model = create_model(np.shape(list(train_ds.take(1))[0][0])[1:], sum([1 for _ in os.listdir(directory)]))
    model.compile()
    model.fit(train_dataset,
              test_dataset,
              batch_size,
              callbacks=keras.callbacks.EarlyStopping())
    evaluate_model(model, val_dataset)
