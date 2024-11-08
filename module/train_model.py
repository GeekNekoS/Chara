from module.project_logging import setup_logger
from module.load_train_test_val import load_train_test_val
from module.create_model import create_model
from module.evaluate_model import evaluate_model
import tensorflow as tf
import keras


logger = setup_logger("train_model")


def train_model(directory: str,
                batch_size: int,
                image_size: tuple[int, int],
                num_classes: int = 58,
                epochs: int = 10):
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
      num_classes : int
          Количество классов для распознавания

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
        train_dataset, test_dataset = load_train_test_val(directory, batch_size, image_size)

        model = create_model(input_shape=image_size + (3,), num_classes=num_classes)
        # Показать модель
        model.summary()
        metrics = [
            keras.metrics.Accuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )
        # Определите колбек для сохранения модели
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/model.keras',  # Путь для сохранения модели
            save_weights_only=False,  # Сохранять всю модель, а не только веса
            save_best_only=True,  # Сохранять только лучшую модель (по метрике)
            monitor='val_accuracy',  # Мониторить метрику (например, валидационную потерю)
            mode='max',  # Сохранять, если метрика уменьшается
            verbose=1  # Логирование процесса
        )

        # Определите колбек для ранней остановки
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',  # Мониторим валидационную потерю
            patience=30,  # Количество эпох без улучшения перед остановкой
            verbose=1,  # Логирование ранней остановки
            restore_best_weights=True  # Восстановить лучшие веса после остановки
        )
        callbacks = [early_stopping_callback, checkpoint_callback]
        history = model.fit(x=train_dataset,
                  batch_size=128,
                  callbacks=callbacks,
                  validation_data=test_dataset,
                  verbose=2,
                  epochs=epochs
                  )
        evaluate_model(model, history, test_dataset)
        logger.info("training completed successfully")
    except Exception as exc:
        logger.error(f"An error occurred during training: {exc}")
