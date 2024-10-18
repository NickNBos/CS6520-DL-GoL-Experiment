from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import trange

from constants import BATCH_SIZE, MAX_PERIOD, NUM_EPOCHS, WORLD_SIZE
from import_data import CATEGORY_NAMES, TOP_15_NAMES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def encode_labels(batch):
    # Find the one-hot encoding for the labels in each class, then
    # concatente into a single vector.
    category = batch['category'].to(torch.int64)
    pattern_id = batch['pattern_id'].to(torch.int64)
    period = batch['period'].nan_to_num(0).to(torch.int64)
    return torch.cat((
        one_hot(category, len(CATEGORY_NAMES)),
        one_hot(pattern_id, len(TOP_15_NAMES)),
        one_hot(period, MAX_PERIOD)),
        dim=2
    ).squeeze().to(torch.float32)

def decode_labels(encoding):
    # Split up the output vector into three parts for the three
    # classification tasks.
    category, encoding = (
        encoding[:, :len(CATEGORY_NAMES)],
        encoding[:, len(CATEGORY_NAMES):])
    pattern, period = (
        encoding[:, :len(TOP_15_NAMES)],
        encoding[:, len(TOP_15_NAMES):])
    assert(period.shape[1] == MAX_PERIOD)

    # Find the predicted most likely category for each classification task.
    return (torch.argmax(category, dim=1),
            torch.argmax(pattern, dim=1),
            torch.argmax(period, dim=1))

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train for the specified number of epochs
    progress = trange(NUM_EPOCHS * (len(train_data) + len(validate_data)))
    train_loss = np.zeros(NUM_EPOCHS)
    validate_loss = np.zeros(NUM_EPOCHS)
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
    # Setup for counting samples
    category_correct = {index: 0 for index in range(len(CATEGORY_NAMES))}
    pattern_id_correct = {index: 0 for index in range(len(TOP_15_NAMES))}
    period_correct = {index: 0 for index in range(MAX_PERIOD)}
    category_count = {index: 0 for index in range(len(CATEGORY_NAMES))}
    pattern_id_count = {index: 0 for index in range(len(TOP_15_NAMES))}
    period_count = {index: 0 for index in range(MAX_PERIOD)}

    # Go through the test data, run them through the model, and count up
    # the number of correct / total predictions.
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = model.to(DEVICE)
    for batch in loader:
        # Grab the test data and predicted labels.
        batch_size = batch['initial_state'].shape[0]
        initial_state = batch['initial_state'].reshape(
            (batch_size, 1, WORLD_SIZE, WORLD_SIZE)
        ).to(torch.float32).to(DEVICE)
        actual_category = batch['category']
        actual_pattern_id = batch['pattern_id']
        actual_period = batch['period'].nan_to_num(0)
        predicted_category, predicted_pattern_id, predicted_period = (
            decode_labels(model.forward(initial_state).cpu()))

        # Break down batch data by classification task and label.
        # TODO: Maybe collect some samples of failures to debug our models?
        for predicted, actual in zip(predicted_category, actual_category):
            if predicted == actual:
                category_correct[int(actual)] += 1
            category_count[int(actual)] += 1
        for predicted, actual in zip(predicted_pattern_id, actual_pattern_id):
            if predicted == actual:
                pattern_id_correct[int(actual)] += 1
            pattern_id_count[int(actual)] += 1
        for predicted, actual in zip(predicted_period, actual_period):
            if predicted == actual:
                period_correct[int(actual)] += 1
            period_count[int(actual)] += 1

    # Print a summary of category predictions.
    category_accuracy = sum(category_correct.values()) / sum(category_count.values())
    print()
    print(f'Category accuracy: {100 * category_accuracy:.2f}%')
    for cat_value, cat_name in enumerate(CATEGORY_NAMES):
        correct, count = category_correct[cat_value], category_count[cat_value]
        accuracy = correct / count
        print(f'{cat_name:>17}: {correct:>4} of {count:>4} ({100 * accuracy:.2f}%)')

    # Print a summary of recognizing the top 15.
    pattern_id_accuracy = sum(pattern_id_correct.values()) / sum(pattern_id_count.values())
    print()
    print(f'  Top-15 accuracy: {100 * pattern_id_accuracy:.2f}%')
    for pattern_id, pattern_name in enumerate(TOP_15_NAMES):
        correct, count = pattern_id_correct[pattern_id], pattern_id_count[pattern_id]
        accuracy = correct / count
        print(f'{pattern_name:>17}: {correct:>4} of {count:>4} ({100 * accuracy:.2f}%)')

    # Print a summary of period predictions.
    period_accuracy = sum(period_correct.values()) / sum(period_count.values())
    print()
    print(f'  Period accuracy: {100 * period_accuracy:.2f}%')
    for period in range(MAX_PERIOD):
        correct, count = period_correct[period], period_count[period]
        if count == 0:
            continue
        accuracy = correct / count
        period = 'n/a' if period < 1 else period
        print(f'{period:>17}: {correct:>4} of {count:>4} ({100 * accuracy:.2f}%)')

