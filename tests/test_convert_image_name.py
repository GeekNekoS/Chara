import os
from PIL import Image
from pathlib import Path
import pytest
from module.convert_image_name import convert_image_name_to_jpg


@pytest.fixture

def create_test_folder(tmpdir):
    """Создание тестовой папки с изображениями для проверки."""
    test_folder = tmpdir.mkdir("test_images")
            
    subfolder1 = test_folder.mkdir("subfolder1")
    subfolder2 = test_folder.mkdir("subfolder2")
    #Создание тестовых изображений
    with open(os.path.join(str(subfolder1), "image1.png"), "wb") as f:
        f.write(b"Test image data")
    with open(os.path.join(str(subfolder2), "image2.bmp"), "wb") as f:
        f.write(b"Test image data")
    print("Успешно.")

    return str(test_folder)

def test_convert_image_name_to_jpg_with_subfolders(create_test_folder):
    """Проверка конвертации изображений в подпапках."""
    convert_image_name_to_jpg(create_test_folder)
    # Проверка создания подпапок для конвертированных изображений
    assert os.path.exists(os.path.join(create_test_folder, "converted_images", "subfolder1"))
    assert os.path.exists(os.path.join(create_test_folder, "converted_images", "subfolder2"))
    
    # Проверка формата jpg
    assert os.path.exists(os.path.join(create_test_folder, "converted_images", "subfolder1", "image1.jpg"))
    assert os.path.exists(os.path.join(create_test_folder, "converted_images", "subfolder2", "image2.jpg"))

    print("Успешно.")

def test_convert_image_name_to_jpg_with_invalid_file_format(create_test_folder):
    """Проверка обработки неверных файлов."""
    # Создание файла, не являющегося изображением
    with open(os.path.join(create_test_folder, "subfolder1", "invalid.txt"), "w") as f:
        f.write("Test text")

    convert_image_name_to_jpg(create_test_folder)
    # Проверка результата конвертации файла, не являющегося изображением
    assert not os.path.exists(os.path.join(create_test_folder, "converted_images", "subfolder1", "invalid.txt"))

    print("Успешно.")

@pytest.mark.usefixtures("my_original_fixture")

def test_simple(my_original_fixture):
    print(my_original_fixture)

test_simple(test_convert_image_name_to_jpg_with_subfolders)
test_simple(test_convert_image_name_to_jpg_with_invalid_file_format)
