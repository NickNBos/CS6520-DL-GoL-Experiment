import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from constants import MAX_PERIOD, VARIATIONS_PER_IMAGE, VIDEO_LEN, WORLD_SIZE
from decode import add_padding, decode
from import_data import CATEGORY_NAMES, TOP_15_NAMES, load_catagolue_data
from simulate import simulate_one


def random_transform(seed, obj, obj_data):
    # Choose random numbers deterministically so we apply the same transforms
    # every time we load an image.
    rng = np.random.default_rng(seed)
    def coin_flip():
        return bool(rng.integers(2))

    # Flip
    if coin_flip():
        obj = np.flip(obj, axis=0)
    if coin_flip():
        obj = np.flip(obj, axis=1)

    # Rotate
    obj = np.rot90(obj, k=rng.integers(4))

    # Translate
    obj = np.roll(obj, rng.integers(WORLD_SIZE), axis=0)
    obj = np.roll(obj, rng.integers(WORLD_SIZE), axis=1)

    # Torch needs the object in a contiguous block of memory.
    obj = np.ascontiguousarray(obj)

    # Step simulation
    period = obj_data.select('period').item()
    if period > 1:
        steps = rng.integers(period)
        if steps > 0:
            # Simulate for some number of steps, then grab the last frame.
            obj = simulate_one(obj, steps)[-1]

    return obj


class GameOfLifeDataset(Dataset):
    def __init__(self):
        self.df = load_catagolue_data().filter(
            (pl.col('period').is_null()) | (pl.col('period') <= MAX_PERIOD)
        )

    def __len__(self):
        return len(self.df) * VARIATIONS_PER_IMAGE

    def __getitem__(self, i):
        # Actually lookup and augment the object data to simulate.
        obj_data = self.df[i // VARIATIONS_PER_IMAGE]
        # TODO: It would probably be better to pre-render and compute sizes for
        # all the objects to better support compositions when we get to video
        # segmentation later, and to avoid repeated computation.
        obj = add_padding(decode(obj_data.select('apgcode').item()))
        obj = random_transform(i, obj, obj_data)

        # Return samples from the Dataset for training.
        return {
            # Initial state of the Game of Life simulation to classify.
            # NOTE: To work with video data, use a DataLoader to fetch a whole
            # batch of initial states, then simulate them in one shot before
            # passing them in as training data.
            'initial_state': obj,
            # What kind of object is this (still_life, oscillator, spaceship,
            # or other)?
            'category': obj_data.select('category').to_numpy().flatten(),
            # What is the period of this object?
            'period': obj_data.select('period').to_numpy().flatten(),
            # Which object is this (one of the top 15 or 'other')
            'obj_id': obj_data.select('top_15').to_numpy().flatten(),
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = GameOfLifeDataset()
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    for batch in dataloader:
        for i, (x, y) in enumerate(np.ndindex(5, 5)):
            plt.subplot(5, 5, i + 1)
            plt.imshow(batch['initial_state'][i].squeeze(), cmap='Greys')
            category = CATEGORY_NAMES[batch['category'][i]]
            period = batch['period'][i].item()
            name = TOP_15_NAMES[batch['obj_id'][i]]
            ax = plt.gca()
            ax.set_title(f'{name}, {category}, {period}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.show()
