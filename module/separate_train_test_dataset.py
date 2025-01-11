import tensorflow as tf
import os
import shutil


def separate_train_test_dataset(dataset_path: str, validation_split: float, train_path: str, test_path: str) -> None:
    for directory in os.listdir(dataset_path):
        cur_original_path = f'{dataset_path}/{directory}'
        cur_train_path = f'{train_path}/{directory}'
        cur_test_path = f'{test_path}/{directory}'
        os.mkdir(cur_train_path)
        os.mkdir(cur_test_path)
        n = len(os.listdir(f'{dataset_path}/{directory}'))
        train_size = int(n * (1 - validation_split))
        for i, name in enumerate(os.listdir(f'{dataset_path}/{directory}')):
            if i < train_size:
                shutil.copy(f'{cur_original_path}/{name}', f'{cur_train_path}/{name}')
            else:
                shutil.copy(f'{cur_original_path}/{name}', f'{cur_test_path}/{name}')