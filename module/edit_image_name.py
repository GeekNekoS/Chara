import os
import multiprocessing as mp


def process_folder(folder_path: str): #вспомогательная функция, которая вызывается отдельно для каждой папки с фото
    folder_counter = 1
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            extension = os.path.splitext(file_name)[1]
            new_file_name = f"image_{folder_counter}{extension}"
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            folder_counter += 1



def edit_image_name(datadir: str): #основная функция для переименования файлов
    if not os.path.exists(datadir):
        print(f"Директория {datadir} не существует.")
        return None

    #folder_counter = {}
    folders = []

    for folder_number in range(1, 129):
        folder_path = os.path.join(datadir, str(folder_number))

        if os.path.exists(folder_path):

            folders.append(folder_path)

            with mp.Pool(processes = os.cpu_count()) as pool:
                pool.map(process_folder, folders)

            #for file_name in os.listdir(folder_path):
            #    file_path = os.path.join(folder_path, file_name)
            #    if os.path.isfile(file_path):

            #        extension = os.path.splitext(file_name)[1]

            #        if folder_path not in folder_counter:
            #            folder_counter[folder_path] = 1

            #        new_file_name = f"image_{folder_counter[folder_path]}{extension}"
            #        new_file_path = os.path.join(folder_path, new_file_name)

            #        os.rename(file_path, new_file_path)

            #        folder_counter[folder_path] += 1
