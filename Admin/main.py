import network
import dataset_loader

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

WIDTH = 64
HEIGHT = 64
NEED_TO_CREATE_DATASET = False

if NEED_TO_CREATE_DATASET:
    dataset_loader.create_dataset()

train_set, train_labels, test_set, test_labels, val_set, val_labels, BATCH_SIZE = dataset_loader.load_dataset()

TRAINING_SIZE = len(train_set) * BATCH_SIZE
TESTING_SIZE = len(test_set) * BATCH_SIZE

EPOCHS = 5
LEARNING_RATE = 0.001

network = network.Network().to(device)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

training_losses = []
training_accuracies = []

testing_losses = []
testing_accuracies = []

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train():
    network.train()
    correct_in_episode = 0
    episode_loss = 0

    for index, images in enumerate(train_set):
        labels = train_labels[index]

        predictions = network(images)
        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()
        correct_in_episode += get_num_correct(predictions, labels)

    training_losses.append(episode_loss)
    training_accuracies.append(correct_in_episode * 100 / TRAINING_SIZE)
    print(f"Epoch: {epoch + 1} accuracy: {correct_in_episode * 100 / TRAINING_SIZE:.2f} loss: {episode_loss:.3f}", end="\t")


def test():
    network.eval()
    episode_loss = 0
    correct_in_episode = 0

    with torch.no_grad():
        for index, images in enumerate(test_set):
            labels = test_labels[index]

            predictions = network(images)
            loss = F.cross_entropy(predictions, labels.long())

            episode_loss = loss.item()
            correct_in_episode += get_num_correct(predictions, labels)

    testing_losses.append(episode_loss)
    testing_accuracies.append(correct_in_episode * 100 / TESTING_SIZE)
    print(f'Validation: Accuracy: {correct_in_episode * 100 / TESTING_SIZE:.2f} loss: {episode_loss:.3f}')

for epoch in range(EPOCHS):
    train()
    test()

plt.plot(list(range(1, len(training_losses)+1)), training_losses, color='blue')
plt.plot(list(range(1, len(testing_losses)+1)), testing_losses, color='red')

plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Loss')

plt.show()

plt.plot(list(range(1, len(training_accuracies)+1)), training_accuracies, color='blue')
plt.plot(list(range(1, len(testing_accuracies)+1)), testing_accuracies, color='red')

plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Accuracy')

plt.ylim(0, 100)
plt.show()