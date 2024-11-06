from project_logging import setup_logger
import os
from PIL import Image
from pathlib import Path


logger = setup_logger("convert_image_name")
def convert_image_name(source_dir: str, target_dir: str) -> None:
    """
    Аргументы:
    source_dir (str): Путь к директории с изображениями.
    target_dir (str): Путь к новой директории с изображениями.
    Вывод:
    None
    """
    logger.info(f"start converting with source_dir: {source_dir}")
    try:
        # Указываем имя папки, в которую будем складывать конвертированные изображения, и путь к ней
        target_dir = os.path.join(Path(source_dir).parent, target_dir)

        # Создаем целевую директорию, если она не существует
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Целевая директория для сохранения: {target_dir}")

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

                                logger.info(f"файл '{filename}' обработан и сохранен как '{new_filename}'")
                        except Exception as e:
                            logger.error(f"Ошибка при обработке файла '{filename}': {str(e)}")

                    else:
                        logger.warning(f"Пропущен файл: {filename} (не изображение или не найден)")
            else:
                logger.warning(f"Пропущена папка: {folder_name} (не является папкой)")
        logger.info("Обработка завершена")
    except Exception as exc:
        logger.error(f"Ошибка при конвертировании изображений: {str(exc)}")

