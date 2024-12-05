import os
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import Callback
from module.project_logging import setup_logger
from module.load_train_test_val import load_train_test_val
from module.create_model import create_model
from module.evaluate_model import evaluate_model
import mlflow


logger = setup_logger("train_model")


def train_model(train_dataset,
                test_dataset,
                batch_size: int,
                image_size: tuple[int, int],
                num_classes: int = 58,
                epochs: int = 10,
                learning_rate: float = 1e-3,
                model_name_prefix: str = 'model'  # Добавлен параметр для изменения имени модели
                ):
    """
    Обучает модель и отслеживает параметры и метрики с использованием MLflow.
    """
    logger.info(f"training starts with batch_size: {batch_size}, image_size: {image_size}, num_classes: {num_classes}")
    try:
        model = create_model(input_shape=image_size + (3,), num_classes=num_classes)

        # Показать модель
        model.summary()

        metrics = [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Recall(),
            keras.metrics.Precision(),
            keras.metrics.AUC()
        ]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        # Определите колбек для сохранения модели
        best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/1/model.keras',  # Путь для сохранения модели
            save_weights_only=False,  # Сохранять всю модель, а не только веса
            save_best_only=True,  # Сохранять только лучшую модель (по метрике)
            monitor='val_categorical_accuracy',  # Мониторить метрику (например, валидационную потерю)
            mode='max',  # Сохранять, если метрика уменьшается
            verbose=1  # Логирование процесса
        )
        # saved_model_callback = SavedModelCallback(
        #     save_dir='models/1/model.keras',
        #     monitor='val_categorical_accuracy',
        #     mode='max',
        #     save_best_only=True,
        #     verbose=1
        #
        # )
        # Определите колбек для ранней остановки
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',  # Мониторим валидационную потерю
            patience=20,  # Количество эпох без улучшения перед остановкой
            verbose=1,  # Логирование ранней остановки
            restore_best_weights=True  # Восстановить лучшие веса после остановки
        )

        # Старт сессии в MLflow
        with mlflow.start_run():
            # Логирование параметров
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("image_size", image_size)
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", learning_rate)

            # Логируем метрики
            history = model.fit(x=train_dataset,
                                validation_data=test_dataset,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=1,
                                callbacks=[early_stopping_callback, best_model_callback])

            ## Запись метрик в MLflow
            for epoch in range(epochs):
                mlflow.log_metric("train_loss", history.history['loss'][epoch])
                mlflow.log_metric("train_accuracy", history.history['categorical_accuracy'][epoch])
                mlflow.log_metric("train_recall", history.history['recall'][epoch])
                mlflow.log_metric("train_precision", history.history['precision'][epoch])
                mlflow.log_metric("train_auc", history.history['auc'][epoch])
                if 'val_loss' in history.history:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch])
                    mlflow.log_metric("val_accuracy", history.history['val_categorical_accuracy'][epoch])
                    mlflow.log_metric("val_recall", history.history['val_recall'][epoch])
                    mlflow.log_metric("val_precision", history.history['val_precision'][epoch])
                    mlflow.log_metric("val_auc", history.history['val_auc'][epoch])

            # Генерация нового имени для модели с учетом эпохи или метрики
            model_name = f"{model_name_prefix}_epoch_{epochs}_acc_{history.history['val_categorical_accuracy'][-1]:.4f}"

            # Сохранение модели с уникальным именем в MLflow
            mlflow.keras.log_model(model, model_name)

            # Загружаем лучшую модель для дальнейшей оценки
            model = tf.keras.models.load_model('models/model')
            evaluate_model(model, history, test_dataset)

            logger.info("training completed successfully")
    except Exception as exc:
        logger.error(f"An error occurred during training: {exc}")


class SavedModelCallback(Callback):
    def __init__(self, save_dir, monitor='val_loss', mode='min', save_best_only=True, verbose=1):
        """
        Колбек для сохранения модели в формате SavedModel, только если метрика улучшается.

        :param save_dir: Путь к директории для сохранения модели.
        :param monitor: Метрика для мониторинга (например, 'val_loss').
        :param mode: Режим мониторинга ('min' или 'max').
        :param save_best_only: Если True, сохраняется только лучшая модель.
        :param verbose: Уровень вывода сообщений (0 или 1).
        """
        super(SavedModelCallback, self).__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        # Установка начального значения метрики
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        """
        Сохраняет модель в формате SavedModel, только если метрика улучшается.

        :param epoch: Номер текущей эпохи.
        :param logs: Логирования метрик обучения.
        """
        # Получение текущего значения метрики
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Metric '{self.monitor}' is not available in logs.")
            return

        # Определение, нужно ли сохранять модель
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            if self.verbose > 0:
                print(f"\n Epoch {epoch+1}: {self.monitor} improved to {current:.4f}, saving model to {self.save_dir}/model_epoch_{epoch+1}")

            # Создаем путь для сохранения модели
            # model_save_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}")
            model_save_path = os.path.join(self.save_dir)
            # Сохраняем модель в формате SavedModel
            tf.saved_model.save(self.model, model_save_path)
            print(f"\n Model saved to: {model_save_path}")

        elif self.verbose > 0:
            print(f"\n Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")



