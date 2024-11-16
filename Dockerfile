#FROM tensorflow/serving:latest-gpu
## Устанавливаем рабочую директорию
#WORKDIR /models
#
## Копируем модели в контейнер
#COPY ./models /models
#
## Копируем конфигурационный файл моделей
#COPY models.config /models/models.config
#
## Открываем порт для REST API
#EXPOSE 8501
#
## Устанавливаем переменные окружения
#ENV MODEL_NAME=model
#ENV MODEL_BASE_PATH=/models
#
#ENTRYPOINT ["tensorflow_model_server",
#            "--model_config_file=/models/models.config",
#            "--allow_version_labels_for_unavailable_models=true",
#            "--rest_api_port=8501",
#            "--model_name=${MODEL_NAME}"]
