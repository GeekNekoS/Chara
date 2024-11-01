import os


def edit_image_name(folder_list: list):
    """
    Изменяет названия файлов в указанных папках, начиная с image_1, image_2 и т.д.

    :param folder_list: Список строк с путями к папкам, содержащим изображения
    :return: None
    """
    for folder in folder_list:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(('.jpeg', '.jpg', '.png', '.jfif'))]
            for index, file in enumerate(files, start=1):
                file_extension = os.path.splitext(file)[1]
                new_name = f"image_{index}{file_extension}"
                os.rename(os.path.join(folder, file), os.path.join(folder, new_name))
        else:
            print(f"Папка {folder} не найдена.")


# if name == "main":
#     folder_list = ['C:/Users/logot/PycharmProjects/Chara/module/convertation_module/images']  # Укажи путь к папке, где лежат фотки
#     edit_image_name(folder_list)