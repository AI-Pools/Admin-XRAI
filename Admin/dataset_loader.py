import h5py
import torch
import numpy as np

from PIL import Image
from os import listdir
from sklearn.utils import shuffle

BATCH_SIZE = 64
CREATE_DATASET = False
WIDTH = 64
HEIGHT = 64


def load_set(path: str):
    dataset = []
    for f in listdir(path):
        dataset.append(np.asarray(Image.open(path + f).resize((WIDTH, HEIGHT)).convert('RGB'), dtype=np.int32).reshape(3, WIDTH, HEIGHT))

    return np.array(dataset, dtype=np.int32)


def create_batch(dataset, labels, batch_size):
    dataset_result = []
    labels_result = []
    data_batch = []
    label_batch = []

    for index in range(len(dataset)):
        if len(data_batch) == batch_size:
            dataset_result.append(data_batch)
            labels_result.append(label_batch)
            data_batch = []
            label_batch = []
        else:
            data_batch.append(dataset[index])
            label_batch.append(labels[index])

    return torch.Tensor(dataset_result), torch.Tensor(labels_result)



def create_dataset():
    dataset_file = h5py.File("dataset.hdf5", "w")

    train_normal = load_set("chest_xray/train/NORMAL/")
    dataset_file.create_dataset("train_normal", train_normal.shape, dtype=np.int32, data=train_normal)

    train_pneumonia = load_set("chest_xray/train/PNEUMONIA/")
    dataset_file.create_dataset("train_pneumonia", train_pneumonia.shape, dtype=np.int32, data=train_pneumonia)

    test_normal = load_set("chest_xray/test/NORMAL/")
    dataset_file.create_dataset("test_normal", test_normal.shape, dtype=np.int32, data=test_normal)

    test_pneumonia = load_set("chest_xray/test/PNEUMONIA/")
    dataset_file.create_dataset("test_pneumonia", test_pneumonia.shape, dtype=np.int32, data=test_pneumonia)

    val_normal = load_set("chest_xray/val/NORMAL/")
    dataset_file.create_dataset("val_normal", val_normal.shape, dtype=np.int32, data=val_normal)

    val_pneumonia = load_set("chest_xray/val/PNEUMONIA/")
    dataset_file.create_dataset("val_pneumonia", val_pneumonia.shape, dtype=np.int32, data=val_pneumonia)


def load_dataset():
    dataset = h5py.File('dataset.hdf5', 'r')

    train_set = np.array(list(dataset["train_normal"]) + list(dataset["train_pneumonia"]), dtype=np.int32)
    test_set = np.array(list(dataset["test_normal"]) + list(dataset["test_pneumonia"]), dtype=np.int32)
    val_set = np.array(list(dataset["val_normal"]) + list(dataset["val_pneumonia"]), dtype=np.int32)

    train_labels = [0] * len(dataset["train_normal"]) + [1] * len(dataset["train_pneumonia"])
    test_labels = [0] * len(dataset["test_normal"]) + [1] * len(dataset["test_pneumonia"])
    val_labels = [0] * len(dataset["val_normal"]) + [1] * len(dataset["val_pneumonia"])

    train_set, train_labels = shuffle(np.array(train_set, dtype=np.int32), np.array(train_labels, dtype=np.int32))
    train_set, train_labels = create_batch(train_set, train_labels, BATCH_SIZE)

    test_set, test_labels = shuffle(np.array(test_set, dtype=np.int32), np.array(test_labels, dtype=np.int32))
    test_set, test_labels = create_batch(test_set, test_labels, BATCH_SIZE)

    val_set, val_labels = shuffle(np.array(val_set, dtype=np.int32), np.array(val_labels, dtype=np.int32))
    val_set, val_labels = create_batch(val_set, val_labels, BATCH_SIZE)

    train_labels = train_labels.long()
    test_labels = test_labels.long()
    val_labels = val_labels.long()

    return train_set, train_labels, test_set, test_labels, val_set, val_labels, BATCH_SIZE