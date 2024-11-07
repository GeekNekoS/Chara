from project_logging import setup_logger
from load_train_test_val import load_train_test_val
from create_model import create_model
from evaluate_model import evaluate_model
import numpy as np
import keras
import os

logger = setup_logger("train_model")


def train_model(directory: str,
                batch_size: int,
                image_size: tuple[int, int]):
    """
      Обучает модель на основе изображений из заданной директории, выполняя загрузку данных,
      создание модели, её компиляцию и обучение. Завершает процесс оценкой модели на
      валидационном наборе данных.

      Параметры:
      ----------
      directory : str
          Путь к директории, содержащей изображения, организованные по подкаталогам для каждой
          категории (метки). Директория должна иметь подкаталоги, где каждая папка соответствует
          отдельному классу для классификации.

      batch_size : int
          Размер пакета (batch) для загрузки данных. Задает количество изображений, передаваемых
          в сеть за один шаг, что влияет на скорость и эффективность обучения.

      image_size : tuple[int, int]
          Размер загружаемых изображений в формате (высота, ширина). Все изображения масштабируются
          до этого размера, обеспечивая согласованность данных.

      Возвращаемое значение:
      ----------------------
      None
          Функция не возвращает значений, но записывает информацию в лог. В ходе работы функция
          выводит сообщения о начале и успешном завершении обучения. В случае возникновения ошибок
          информация также записывается в лог.

      Логика работы:
      --------------
      1. **Загрузка данных** — функция `load_train_test_val` загружает данные из `directory`, разбивая
         их на обучающий, тестовый и валидационный наборы данных.
      2. **Создание модели** — функция `create_model` создает сверточную нейронную сеть, подходящую для
         указанной задачи классификации, определяя входную форму и количество классов на основе загруженных
         данных.
      3. **Компиляция модели** — модель компилируется с использованием стандартных параметров.
      4. **Обучение модели** — модель обучается с использованием метода `fit` и колбэка `EarlyStopping`,
         который завершает обучение при отсутствии улучшений, предотвращая переобучение.
      5. **Оценка модели** — модель оценивается на валидационном наборе с использованием функции `evaluate_model`,
         чтобы проверить её точность после обучения.
      6. **Логирование** — начало, успешное завершение и ошибки в процессе обучения записываются в лог.

      Примечание:
      -----------
      Функция `create_model` определяет количество классов на основе количества поддиректорий в `directory`,
      предполагая, что каждая поддиректория представляет один класс.
      """
    logger.info(f"training starts with directory: {directory}, batch_size: {batch_size}, image_size: {image_size}")
    try:
        train_dataset, test_dataset, val_dataset = load_train_test_val(directory, batch_size, image_size)
        model = create_model(np.shape(list(train_dataset.take(1))[0][0]), sum([1 for _ in os.listdir(directory)]))
        model.compile()
        model.fit(train_dataset,
                  test_dataset,
                  batch_size,
                  callbacks=keras.callbacks.EarlyStopping())
        evaluate_model(model, val_dataset)
        logger.info("training completed successfully")
    except Exception as exc:
        logger.error(f"An error occurred during training: {exc}")
