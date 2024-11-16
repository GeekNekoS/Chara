"""
Модуль, делающий прогноз для фото пользователя
"""
import requests
import json


def get_prediction(image_data):
    url = "http://tf_serving:8501/v1/models/my_model:predict"
    headers = {"content-type": "application/json"}

    # Подготовьте данные в формате, который ожидает модель
    data = json.dumps({"instances": [image_data.tolist()]})

    response = requests.post(url, data=data, headers=headers)
    return response.json()

