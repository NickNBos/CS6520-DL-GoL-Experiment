import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import trange

from constants import BATCH_SIZE, NUM_EPOCHS, WORLD_SIZE
from evaluator import Evaluator
from import_data import CATEGORY_NAMES, TOP_15_NAMES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_labels(batch):
    # Find the one-hot encoding for the labels in each class, then
    # concatente into a single vector.
    category = batch['category'].to(torch.int64)
    pattern_id = batch['pattern_id'].to(torch.int64)
    return torch.cat((
        one_hot(category, len(CATEGORY_NAMES)),
        one_hot(pattern_id, len(TOP_15_NAMES))),
        dim=2
    ).squeeze().to(torch.float32)


def decode_labels(encoding):
    # Split up the output vector into three parts for the three
    # classification tasks.
    assert encoding.shape[-1] == len(CATEGORY_NAMES) + len(TOP_15_NAMES)
    category, pattern = (
        encoding[:, :len(CATEGORY_NAMES)],
        encoding[:, -len(TOP_15_NAMES):])

    # Find the predicted most likely category for each classification task.
    return (torch.argmax(category, dim=1),
            torch.argmax(pattern, dim=1))

def chart_loss_curves(train_loss, validate_loss, filename):
    plt.figure()
    plt.plot(np.arange(NUM_EPOCHS), train_loss, label='training loss')
    plt.plot(np.arange(NUM_EPOCHS), validate_loss, label='validation loss')
    plt.legend()
    plt.savefig(filename, dpi=600)


def train_model(model, train_data, validate_data):
    # Prepare to load data
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(
        validate_data, batch_size=BATCH_SIZE, shuffle=True)
    model = model.to(DEVICE)

    # Prepare to optimize model parameters
    # TODO: Investigate other loss criteria, such as soft F1 score.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train for the specified number of epochs
    progress = trange(NUM_EPOCHS * (len(train_data) + len(validate_data)))
    train_loss = np.zeros(NUM_EPOCHS)
    validate_loss = np.zeros(NUM_EPOCHS)
    # TODO: Use an Evaluator to plot precision / recall over epochs
    for epoch in range(NUM_EPOCHS):
        # Run the training data for this epoch throught model, compute loss,
        # and adjust weights after each batch.
        for batch in train_loader:
            batch_size = batch['initial_state'].shape[0]
            initial_state = batch['initial_state'].reshape(
                (batch_size, 1, WORLD_SIZE, WORLD_SIZE)
            ).to(torch.float32).to(DEVICE)
            labels = encode_labels(batch).to(DEVICE)

            optimizer.zero_grad()
            predictions = model.forward(initial_state)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            progress.update(batch_size)
            progress.set_description(f'Loss={loss:.2f}')
        train_loss[epoch] = loss

        # Now go through the validation data to see how well the model performs
        # on that with the weight values we've tuned so far.
        with torch.no_grad():
            for batch in validate_loader:
                batch_size = batch['initial_state'].shape[0]
                initial_state = batch['initial_state'].reshape(
                    (batch_size, 1, WORLD_SIZE, WORLD_SIZE)
                ).to(torch.float32).to(DEVICE)
                labels = encode_labels(batch).to(DEVICE)

                predictions = model.forward(initial_state)
                loss = criterion(predictions, labels)
                progress.update(batch_size)
        validate_loss[epoch] = loss

    # Return loss data over time to visualize training performance.
    return train_loss, validate_loss


def test_model(model, test_data):
    # Go through the test data, run them through the model, and count up
    # the number of correct / total predictions.
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = model.to(DEVICE)
    evaluator = Evaluator()
    for batch in loader:
        # Grab the test data and predicted labels.
        batch_size = batch['initial_state'].shape[0]
        initial_state = batch['initial_state'].reshape(
            (batch_size, 1, WORLD_SIZE, WORLD_SIZE)
        ).to(torch.float32).to(DEVICE)
        true_category = batch['category']
        true_pattern_id = batch['pattern_id']
        pred_category, pred_pattern_id = (
            decode_labels(model.forward(initial_state).cpu()))
        evaluator.ingest_batch(
            true_category, pred_category, true_pattern_id, pred_pattern_id)
    evaluator.print_summary()

