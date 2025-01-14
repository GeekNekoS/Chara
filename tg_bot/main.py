import asyncio
import os
import cv2
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType, FSInputFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from results_handler import handle_results
from ultralytics import YOLO


# Замените на токен вашего бота
API_TOKEN = "8100688188:AAH99gxLH0qZlv3AdQc4GmfAMWhkhvDLaoc"

# Пусть к модели Keras
MODEL_PATH = "../models/3/model.keras"

# Пусть к классификатору Haar Cascade для обнаружения лиц
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Загрузка модели и классификатора
# model = load_model(MODEL_PATH)
model = YOLO('../models/4/best (1).pt')
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# Функция для обработки изображения
def preprocess_image(image):
    image = Image.fromarray(image).convert("RGB")
    image = image.resize((64, 64))  # Измените размер в соответствии с моделью
    image_array = np.array(image) / 255.0  # Нормализация
    image_array = np.expand_dims(image_array, axis=0)  # Добавление размерности
    return image_array


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message):
    # Получаем информацию о файле фото
    photo = message.photo[-1]  # Берем фото с наивысшим разрешением
    file_info = await bot.get_file(photo.file_id)
    file_path = f"downloads/{photo.file_id}.jpg"

    try:
        # Скачиваем фото
        await bot.download_file(file_info.file_path, destination=file_path)

        # Загружаем фото и переводим в grayscale для обнаружения лиц
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Обнаруживаем лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            await message.answer("Лица не обнаружены на изображении.")
        else:
            for i, (x, y, w, h) in enumerate(faces):
                # Извлекаем лицо
                face = image[y:y + h, x:x + w]

                # Предобрабатываем лицо и передаем в модель
                # input_data = preprocess_image(face)
                # prediction = model.predict(input_data)
                prediction = model(face)
                prediction = [(prediction[0].probs.top5[i], prediction[0].probs.top5conf[i]) for i in range(len(prediction[0].probs.top5))]
                # prediction = sorted(enumerate(prediction[0]), key=lambda x: x[1], reverse=True)[:5]
                # print(prediction)
                predictions_with_probabilities = handle_results(prediction)

                # Сохраняем лицо как временное изображение
                face_path = f"downloads/face_{photo.file_id}_{i}.jpg"
                cv2.imwrite(face_path, face)

                # Отправляем лицо с предсказанием
                input_file = FSInputFile(face_path)
                await message.answer_photo(input_file, caption=f"Результат модели:\n{predictions_with_probabilities}")

                # Удаляем временное изображение
                os.remove(face_path)

        # Удаляем оригинальное изображение
        os.remove(file_path)
    except Exception as e:
        await message.answer(f"Ошибка при обработке изображения: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)


@dp.message(F.text == "/start")
async def send_welcome(message: Message):
    await message.answer("Привет! Отправь мне фото, и я обработаю его с помощью нейросети.")


async def main():
    # Создаем папку для временных файлов, если она отсутствует
    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    print("Бот запущен и готов к работе!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
