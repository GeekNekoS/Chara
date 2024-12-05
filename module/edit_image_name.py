from module.project_logging import setup_logger
import os
from multiprocessing import Pool, cpu_count


logger = setup_logger("edit_image_name")


def rename_files_in_folder(folder_path: str):
    """
    Функция для переименования файлов в одной папке.
    """
    try:
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} does not exist")
            return

        folder_counter = 1
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                extension = os.path.splitext(file_name)[1]
                new_file_name = f"image_{folder_counter}{extension}"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(file_path, new_file_path)
                folder_counter += 1

        logger.info(f"Renaming completed in folder: {folder_path}")
    except Exception as exc:
        logger.error(f"Error renaming files in {folder_path}: {exc}")

def edit_image_name(datadir: str):
    """
    Основная функция для обработки всех папок в каталоге `datadir` с использованием multiprocessing.
    """
    logger.info(f"Editing image names starts with dir: {datadir}")
    try:
        if not os.path.exists(datadir):
            logger.warning(f"{datadir} does not exist")
            return

        # Получаем список всех папок в `datadir`
        all_folders = [os.path.join(datadir, folder) for folder in os.listdir(datadir)
                       if os.path.isdir(os.path.join(datadir, folder))]

        if not all_folders:
            logger.warning("No folders found in the provided directory.")
            return

        # Определяем количество доступных ядер
        num_workers = min(cpu_count(), len(all_folders))
        logger.info(f"Using {num_workers} parallel workers")

        # Запускаем параллельную обработку папок
        with Pool(processes=num_workers) as pool:
            pool.map(rename_files_in_folder, all_folders)

        logger.info("Operation successfully completed")
    except Exception as exc:
        logger.error(f"Error during image renaming: {exc}")