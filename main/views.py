from django.shortcuts import render
from django.views import View
from .forms import ImageForm
import requests
from django.core.files.storage import default_storage
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from user_photo.settings import MEDIA_URL


class IndexView(View):
    def get(self, request):
        return render(request, 'main/index.html')


class PhotoView(View):
    def get(self, request):
        form = ImageForm()
        return render(request, 'main/photo.html', {'form': form})


class ImageUploadView(View):
    """Обработка изображений, загруженных пользователями"""

    def get(self, request):
        form = ImageForm()
        return render(request, 'main/photo.html', {'form': form})

    def post(self, request):
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            uploaded_file = form.save()  # Сохраняем загруженный файл
            file_path = default_storage.path(uploaded_file.image.path)
            # Отправляем на инференс сервер и получаем ответ
            prediction_result = self.handle_uploaded_file(file_path)
            prediction_result['image'] = img_obj
            # Имя файла в папке media
            filepath_types = "images/person_type.jpg"  # имя файла с типами личности
            # Формирование URL к фото
            image_url = f"{MEDIA_URL}{filepath_types}"
            prediction_result['image_types_url'] = image_url

            # Передаем результат в шаблон
            return render(request, 'main/result.html', {'result': prediction_result})

        return render(request, 'main/photo.html', {'form': form})

    def handle_uploaded_file(self, file_path):
        # Загружаем изображение
        img = Image.open(file_path)

        # Изменяем размер изображения в соответствии с требованиями модели
        img_resized = img.resize((64, 64))  # Пример: размер 224x224 для большинства моделей

        # Преобразуем изображение в numpy массив и нормализуем его
        img_array = np.array(img_resized) / 255.0  # Нормализация изображений (если требуется)

        # Проверяем, если изображение имеет 3 канала (RGB), если нет, добавляем их
        if img_array.ndim == 2:  # Черно-белое изображение
            img_array = np.stack([img_array] * 3, axis=-1)  # Преобразуем в 3 канала

        # Добавляем дополнительную размерность для батча
        img_array = np.expand_dims(img_array, axis=0)

        # Получаем переменные окружения для TensorFlow Serving
        TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "flask_api")
        # TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "localhost")
        TF_SERVING_PORT = os.getenv("TF_SERVING_PORT", "5000")

        # Формируем URL для обращения к TensorFlow Serving
        url = f"http://{TF_SERVING_HOST}:{TF_SERVING_PORT}/predict"  # адрес инференс сервера
        headers = {"content-type": "application/json"}

        # Формируем JSON-запрос для отправки
        data = {
                    "input": img_array.tolist(),
        }

        # Отправка запроса на сервер
        response = requests.post(url, json=data, headers=headers)

        # Проверяем успешность запроса
        if response.status_code == 200:
            response_json = response.json()
            predictions = response_json.get("predictions", [])
            if predictions:
                predictions = predictions[0]  # Если есть предсказания
                result = self.process_predictions(predictions)  # Обрабатываем предсказания
                # Преобразуем изображение в строку Base64
                img_base64 = self.encode_image_to_base64(img)
                # Добавляем изображение в результат
                result["image"] = img_base64
                return result
            else:
                return {"error": "No predictions found in response"}
        else:
            return {"error": "Failed to get prediction", "status_code": response.status_code, "details": response.text}

    def process_predictions(self, predictions):
        # Находим индекс максимальной вероятности и саму вероятность
        predicted_class = np.argmax(predictions)
        # print(predictions)
        confidence = predictions[predicted_class]
        result = {}
        result["confidence"] = round(confidence * 100, 2)
        # Пример отображения информации о классе
        result["class_name"] = self.get_class_name(predicted_class)
        # Преобразуем результат в читаемый для пользователя вид
        return result

    @staticmethod
    def get_class_name(class_index):
        # Здесь можно использовать словарь классов, если у вас есть такое соответствие
        class_names = [f"Психологический тип личности {i}" for i in range(128)]  # Пример имен классов
        return class_names[class_index] if class_index < len(class_names) else "Unknown class"

    @staticmethod
    def encode_image_to_base64(img):
        """
        Функция для преобразования изображения в строку Base64.
        """
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64


