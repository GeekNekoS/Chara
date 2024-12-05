from module.project_logging import setup_logger
import os
from PIL import Image
from pathlib import Path
from multiprocessing import Pool


logger = setup_logger("convert_image_name")


# Вспомогательная функция
def process_folder(folder_path: str, target_dir: str):
    """
    Обрабатывает одну папку: конвертирует изображения в формат JPEG и сохраняет в целевую директорию.
    """
    logger.info(f"Начало обработки папки: {folder_path}")
    try:
        new_folder_path = os.path.join(target_dir, os.path.basename(folder_path))
        os.makedirs(new_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        new_filename = f"{Path(filename).stem}.jpg"
                        new_file_path = os.path.join(new_folder_path, new_filename)

                        # Конвертируем изображение в формат RGB и сохраняем
                        rgb_img = img.convert("RGB")
                        rgb_img.save(new_file_path, "JPEG")
                        logger.info(f"Файл '{filename}' обработан и сохранен как '{new_filename}'")
                except Exception as e:
                    logger.error(f"Ошибка при обработке файла '{filename}' в папке '{folder_path}': {str(e)}")
            else:
                logger.warning(f"Пропущен файл: {filename} (не изображение или не найден)")

        logger.info(f"Обработка папки завершена: {folder_path}")
    except Exception as e:
        logger.error(f"Ошибка при обработке папки '{folder_path}': {str(e)}")


# Основная функция для конвертации
def convert_image_name(source_dir: str, target_dir: str) -> None:
    """
    Конвертирует все изображения в формате папок из `source_dir` в JPEG и сохраняет в `target_dir`.

    Аргументы:
    - source_dir (str): Путь к директории с изображениями.
    - target_dir (str): Путь к новой директории для сохранения изображений.
    """
    logger.info(f"Начало конвертации изображений. Исходная директория: {source_dir}")
    try:
        target_dir = os.path.join(Path(source_dir).parent, target_dir)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Целевая директория для сохранения: {target_dir}")

        # Список всех папок в исходной директории
        folders = [os.path.join(source_dir, folder_name) for folder_name in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, folder_name))]

        # Используем multiprocessing для обработки каждой папки
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap(process_folder, [(folder, target_dir) for folder in folders])

        logger.info("Конвертация изображений завершена.")
    except Exception as exc:
        logger.error(f"Ошибка при конвертировании изображений: {str(exc)}")

