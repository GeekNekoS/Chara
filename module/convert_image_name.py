from project_logging import logger
import os
from PIL import Image
from pathlib import Path


def convert_image_name(source_dir: str):
    """
    Аргументы:
    nameDir (str): Путь к директории с изображениями.
    Вывод:
    None
    """
    logger.info(f"convert_image_name: start with source_dir: {source_dir}")
    try:
        # Указываем имя папки, в которую будем складывать конвертированные изображения, и путь к ней
        target_dir_name = "converted_images" # имя новой папки с конвертированными изображениями
        target_dir = os.path.join(Path(source_dir).parent, target_dir_name)

        # Создаем целевую директорию, если она не существует
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"convert_image_name: Целевая директория для сохранения: {target_dir}")

        # Перебираем все папки в исходной директории
        for folder_name in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder_name)

            # Проверяем, является ли папка директорией
            if os.path.isdir(folder_path):
                # Создаем новую папку в целевой директории с таким же именем
                new_folder_path = os.path.join(target_dir, folder_name)
                os.makedirs(new_folder_path, exist_ok=True)

                # Перебираем все файлы в текущей папке
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)

                # Проверяем, является ли файл изображением
                    if os.path.isfile(file_path):
                        try:
                            # Открываем изображение
                            with Image.open(file_path) as img:
                            # Преобразуем в RGB и сохраняем в новом формате
                                new_filename = f"{Path(filename).stem}.jpg"
                                new_file_path = os.path.join(new_folder_path, new_filename)

                                # Сохраняем изображение в формате JPEG
                                rgb_img = img.convert("RGB")
                                rgb_img.save(new_file_path, "JPEG")

                                logger.info(f"convert_image_name: файл '{filename}' обработан и сохранен как '{new_filename}'")
                        except Exception as e:
                            logger.error(f"convert_image_name: Ошибка при обработке файла '{filename}': {str(e)}")

                    else:
                        logger.warning(f"convert_image_name: Пропущен файл: {filename} (не изображение или не найден)")
            else:
                logger.warning(f"convert_image_name: Пропущена папка: {folder_name} (не является папкой)")
        logger.info("convert_image_name: Обработка завершена")
    except Exception as exc:
        logger.error(f"convert_image_name: Ошибка при конвертировании изображений: {str(exc)}")


# Пример использования:
"""
if __name__ == "__main__":
source_dir = "./images/" # путь к исходному каталогу изображений
convert_image_name_to_jpg(source_dir)
"""
