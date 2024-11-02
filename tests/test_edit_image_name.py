import os
import pytest
import tempfile
import shutil

from module.edit_image_name import edit_image_name


def create_test_structure(base_dir, folder_number, files):
    os.makedirs(os.path.join(base_dir, str(folder_number)), exist_ok=True)
    for file_name in files:
        with open(os.path.join(base_dir, str(folder_number), file_name), 'w') as f:
            f.write('test')


def clean_up(test_dir):
    shutil.rmtree(test_dir)


@pytest.fixture
def setup_test_environment():
    temp_dir = tempfile.mkdtemp()

    test_cases = {
        'case_one': ['image1.jpeg', 'image2.jpg', 'image3.png'],
        'case_two': ['picture1.jfif', 'picture2.jfif'],
        'case_three': ['file1.png', 'file2.png', 'file3.png'],
        'case_four': ['img1.jpeg', 'img2.jpg', 'img3.png', 'img4.jfif'],
        'case_five': ['snapshot1.png', 'snapshot2.png'],
    }

    for folder_num, (case_name, files) in enumerate(test_cases.items(), start=1):
        create_test_structure(temp_dir, folder_num, files)

    yield temp_dir

    clean_up(temp_dir)


def test_image_renaming_case_one(setup_test_environment):
    datadir = setup_test_environment
    edit_image_name(datadir)

    for folder_number in range(1, 6):
        folder_path = os.path.join(datadir, str(folder_number))
        renamed_files = os.listdir(folder_path)

        for i, file in enumerate(sorted(renamed_files), start=1):
            _, extension = os.path.splitext(file)
            assert file == f"image_{i}{extension}", f"{file} не соответствует формату image_{i}{extension}"

        assert renamed_files[0].startswith("image_1"), f"В папке {folder_number} нумерация не начинается с 1."


def test_image_renaming_case_two(setup_test_environment):
    datadir = setup_test_environment
    edit_image_name(datadir)

    for folder_number in range(1, 6):
        folder_path = os.path.join(datadir, str(folder_number))
        renamed_files = os.listdir(folder_path)

        for i, file in enumerate(sorted(renamed_files), start=1):
            _, extension = os.path.splitext(file)
            assert file == f"image_{i}{extension}"

        assert renamed_files[0].startswith("image_1")


def test_image_renaming_case_three(setup_test_environment):
    datadir = setup_test_environment
    edit_image_name(datadir)

    for folder_number in range(1, 6):
        folder_path = os.path.join(datadir, str(folder_number))
        renamed_files = os.listdir(folder_path)

        for i, file in enumerate(sorted(renamed_files), start=1):
            _, extension = os.path.splitext(file)
            assert file == f"image_{i}{extension}"

        assert renamed_files[0].startswith("image_1")


def test_image_renaming_case_four(setup_test_environment):
    datadir = setup_test_environment
    edit_image_name(datadir)

    for folder_number in range(1, 6):
        folder_path = os.path.join(datadir, str(folder_number))
        renamed_files = os.listdir(folder_path)

        for i, file in enumerate(sorted(renamed_files), start=1):
            _, extension = os.path.splitext(file)
            assert file == f"image_{i}{extension}"

        assert renamed_files[0].startswith("image_1")


def test_image_renaming_case_five(setup_test_environment):
    datadir = setup_test_environment
    edit_image_name(datadir)

    for folder_number in range(1, 6):
        folder_path = os.path.join(datadir, str(folder_number))
        renamed_files = os.listdir(folder_path)

        for i, file in enumerate(sorted(renamed_files), start=1):
            _, extension = os.path.splitext(file)
            assert file == f"image_{i}{extension}"

        assert renamed_files[0].startswith("image_1")
