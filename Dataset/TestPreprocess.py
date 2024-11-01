from Preprocess import load_and_split_dataset


train_dataset, test_dataset, val_dataset = load_and_split_dataset('TestDataset', 8, (224, 224))
print(train_dataset)
print(test_dataset)
print(val_dataset)
