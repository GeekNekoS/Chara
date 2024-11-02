import os
from PIL import Image
from pathlib import Path

def convert_image_name_to_jpg(source_dir:str):
    """
    Аргументы:
        nameDir (str): Путь к директории с изображениями.
    Вывод:
        None
    """
    # Указываем имя папки, в которую будем складывать конвертированные изображения, и путь к ней
    target_dir_name = "converted_images"  # имя новой папки с конвертированными изображениями
    target_dir = os.path.join(Path(source_dir).parent, target_dir_name)
    
    # Создаем целевую директорию, если она не существует
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"Целевая директория для сохранения: {target_dir}")
    
    # Перебираем все файлы в исходной директории с изображениями
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)  # Полный путь к исходным вайлам
        #print(f"Обработка файла: {file_path}")

        # Проверяем, является ли файл изображением
        if os.path.isfile(file_path):
            try:
                # Открываем изображение
                with Image.open(file_path) as img:
                    # Проверка формата для отладки
                    #print(f"Формат изображения '{filename}': {img.format}")   
                    # Преобразуем в RGB (для корректного сохранения в JPEG) и сохраняем в новом формате
                    new_filename = f"{Path(filename).stem}.jpg"
                    new_file_path = os.path.join(target_dir, new_filename)
                    
                    # Сохраняем изображение в формате JPEG
                    rgb_img = img.convert("RGB")
                    rgb_img.save(new_file_path, "JPEG")
                    
                    #print(f"Файл '{filename}' успешно конвертирован и сохранен как '{new_filename}'")
            except Exception as e:
                print(f"Ошибка при обработке файла '{filename}': {e}")

        else:
            print(f"Пропущен файл: {filename} (не является изображением или не найден)")
    print("Обработка завершена")        

# Пример использования функции
#source_directory = "/Users/zenecka/Desktop/ИРИТ РТФ/3 семестр/ПП 3 сем/Chara/module/images" # путь к исходному каталогу изображений
#convert_image_name_to_jpg(source_directory)
