import os
import multiprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Параметры
source_dir = '/mnt/d/project_practicum/dataset'  # Путь к исходной папке с изображениями
destination_dir = '/mnt/d/project_practicum/dataset_augmented'  # Путь к папке для сохранения аугментированных изображений
batch_size = 32  # Размер пакета для генератора
target_size = (256, 256)  # Размер изображений после аугментации


# Функция для обработки одной категории
def process_category(category):
    # Путь к категории и папке назначения
    category_path = os.path.join(source_dir, category)
    category_dest = os.path.join(destination_dir, category)

    # Проверим, существует ли папка назначения, если нет - создадим
    if not os.path.exists(category_dest):
        os.makedirs(category_dest)

    # Создаем объект ImageDataGenerator для аугментации
    datagen = ImageDataGenerator(
        rotation_range=40,  # случайное вращение изображений
        width_shift_range=0.2,  # случайное смещение по ширине
        height_shift_range=0.2,  # случайное смещение по высоте
        shear_range=0.2,  # случайное сдвигание
        zoom_range=0.2,  # случайное увеличение или уменьшение масштаба
        horizontal_flip=True,  # случайное зеркальное отражение
        fill_mode='nearest'  # заполнение пустых мест после трансформаций
    )

    # Создаем генератор для изображений в текущей категории
    flow = datagen.flow_from_directory(
        source_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # или 'categorical', в зависимости от задачи
        save_to_dir=category_dest,
        save_prefix='aug',  # Префикс для имен сохраненных файлов
        save_format='jpeg',  # Формат сохранения изображений
        subset=None,  # Мы будем использовать все изображения
        classes=[category],  # Ограничиваем генерацию только для текущей категории
        shuffle=True,  # перемешиваем изображения
    )

    # Выполняем аугментацию и сохраняем изображения
    for _ in range(50):  # Количество пакетов изображений для аугментации, можно увеличить
        next(flow)


# Главная функция для запуска параллельной обработки
def main():
    # Получаем список категорий (папок) в исходной папке
    categories = [category for category in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, category))]

    # Используем Pool для параллельной обработки категорий на разных ядрах
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_category, categories)

    print("Аугментация завершена!")


if __name__ == "__main__":
    main()
