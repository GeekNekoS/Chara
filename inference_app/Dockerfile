# Используем официальный образ TensorFlow
FROM tensorflow/tensorflow:2.17.0

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы приложения
COPY ./inference_app /app
COPY ./inference_app/requirements.txt /app/requirements.txt
COPY ./models/1/model.keras /app/models/1/model.keras

# Устанавливаем зависимости
RUN pip install --upgrade pip
RUN python -m pip uninstall -y blinker || true
RUN pip install --no-cache-dir --ignore-installed -r /app/requirements.txt

# Указываем порт для приложения
EXPOSE 5000

# Запуск приложения
CMD ["python", "app/app.py"]
