from copy import deepcopy
from math import inf

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from constants import BATCH_SIZE, NUM_EPOCHS, WORLD_SIZE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unpack_batch(batch):
    batch_size = batch['initial_state'].shape[0]
    initial_state = batch['initial_state'].reshape(
        (batch_size, 1, WORLD_SIZE, WORLD_SIZE)
    ).to(torch.float32).to(DEVICE)
    true_category = batch['category'].to(DEVICE)
    true_pattern_id = batch['pattern_id'].to(DEVICE)
    return batch_size, initial_state, true_category, true_pattern_id


def train_model(model, train_data, validate_data, metrics_tracker,
                optimizer_class=optim.SGD):
    # Prepare to load data
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True)
    validate_loader = DataLoader(
        validate_data, batch_size=BATCH_SIZE, shuffle=True)
    model = model.to(DEVICE)

    # Prepare to optimize model parameters
    optimizer = optimizer_class(model.parameters())

    # Keep track of the best model parameters found during training.
    min_loss = inf
    best_state = None

    # Train for the specified number of epochs
    progress = trange(NUM_EPOCHS * (len(train_data) + len(validate_data)))
    for epoch in range(NUM_EPOCHS):
        # Run the training data for this epoch throught model, compute loss,
        # and adjust weights after each batch.
        for batch in train_loader:
            batch_size, initial_state, true_category, true_pattern_id = \
                    unpack_batch(batch)

            optimizer.zero_grad()
            pred_category_oh, pred_pattern_id_oh = model.forward(initial_state)
            loss = metrics_tracker.score_batch(
                true_category, pred_category_oh,
                true_pattern_id, pred_pattern_id_oh, 'train', epoch)
            if loss < min_loss:
                min_loss = float(loss)
                best_state = deepcopy(model.state_dict())
            loss.backward()
            optimizer.step()

            progress.update(batch_size)
            progress.set_description(f'Loss={loss:.2f}')

        # Now go through the validation data to see how well the model performs
        # on that with the weight values we've tuned so far.
        with torch.no_grad():
            for batch in validate_loader:
                batch_size, initial_state, true_category, true_pattern_id = \
                        unpack_batch(batch)
                pred_category_oh, pred_pattern_id_oh = model.forward(initial_state)
                loss = metrics_tracker.score_batch(
                    true_category, pred_category_oh,
                    true_pattern_id, pred_pattern_id_oh, 'validate', epoch)
                progress.update(batch_size)
    return best_state


def test_model(model, test_data, metrics_tracker):
    # Go through the test data, run them through the model, and count up
    # the number of correct / total predictions.
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    model = model.to(DEVICE)
    for batch in loader:
        _, initial_state, true_category, true_pattern_id = unpack_batch(batch)
        pred_category_oh, pred_pattern_id_oh = model.forward(initial_state)
        metrics_tracker.score_batch(
            true_category, pred_category_oh,
            true_pattern_id, pred_pattern_id_oh, 'test')
