from project_logging import setup_logger
import os

logger = setup_logger("edit_image_name")


def edit_image_name(datadir: str):
    logger.info(f"editing img name starts with dir: {datadir}")
    try:
        if not os.path.exists(datadir):
            logger.warning(f"{datadir} does not exist")
            return None

        folder_counter = {}

        for folder_number in range(1, 129):
            folder_path = os.path.join(datadir, str(folder_number))

            if os.path.exists(folder_path):

                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):

                        extension = os.path.splitext(file_name)[1]

                        if folder_path not in folder_counter:
                            folder_counter[folder_path] = 1

                        new_file_name = f"image_{folder_counter[folder_path]}{extension}"
                        new_file_path = os.path.join(folder_path, new_file_name)

                        os.rename(file_path, new_file_path)

                        folder_counter[folder_path] += 1
        logger.info(f"operation successful ended")
    except Exception as exc:
        logger.error(f"Ошибка при изменении имени изображений: {exc}")
