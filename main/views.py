from django.shortcuts import render
from django.views import View
from .forms import ImageForm
import requests
from django.core.files.storage import default_storage
import os
import numpy as np
from PIL import Image


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
            # form.save()
            # img_obj = form.instance
            # return render(request, 'main/photo.html', {'form': form, 'img_obj': img_obj})
            uploaded_file = form.save()  # Сохраняем загруженный файл
            file_path = default_storage.path(uploaded_file.image.path)

            # Отправляем на инференс сервер и получаем ответ
            prediction_result = self.handle_uploaded_file(file_path)

            # Передаем результат в шаблон
            return render(request, 'main/result.html', {'result': prediction_result})

        return render(request, 'main/photo.html', {'form': form})

    @staticmethod
    def handle_uploaded_file(file_path):
        # Загружаем изображение
        img = Image.open(file_path)
        # Изменяем размер изображения
        img_resized = img.resize((128, 128))  # Примерный размер
        # Преобразуем изображение в numpy массив
        img_array = np.array(img_resized)

        TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "localhost")
        TF_SERVING_PORT = os.getenv("TF_SERVING_PORT", "8501")
        # Отправка запроса на инференс сервер
        url = f"http://{TF_SERVING_HOST}:{TF_SERVING_PORT}/v1/models/model:predict"  # адрес инференс сервера
        headers = {"content-type": "application/json"}

        # Добавляем дополнительное измерение для батча
        img_array = np.expand_dims(img_array, axis=0)

        # Формируем JSON-запрос
        data = {
            "signature_name": "serving_default",
            "instances": [
                {
                    "inputs": img_array.tolist(),
                }
            ]
        }

        response = requests.post(url, json=data, headers=headers)
        return response.json()