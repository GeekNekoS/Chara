from module.load_train_test_val import load_train_test_val


def test_load_train_test_val():
    directory_path = 'TestDataset'
    batch_size = 8
    image_size = (224, 224)
    train_dataset, test_dataset, val_dataset = load_train_test_val(directory_path, batch_size, image_size)
    assert train_dataset
    assert test_dataset
    assert val_dataset
