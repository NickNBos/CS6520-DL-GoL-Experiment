import numpy as np
import polars as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from constants import DATASET_SIZE, MAX_PERIOD, VARIATIONS_PER_IMAGE, WORLD_SIZE
from import_data import CATEGORY_NAMES, TOP_15_NAMES, load_catagolue_data
from simulate import simulate_one

SPLIT_RATIO = (70, 15, 15) # train, validate, test


class GameOfLifeDataset(Dataset):
    def __init__(self, df, seed=0):
        self.df = df.with_columns(
            index_threshold=pl.col('multiplier').cum_sum() * VARIATIONS_PER_IMAGE
        )
        self.length = self.df.select('multiplier').sum().item() * VARIATIONS_PER_IMAGE
        self.seed = seed

    def __len__(self):
        return self.length

    def random_initial_state(self, i, pattern_data):
        # Look up the pattern
        pattern = np.array(pattern_data.select('pattern').item().to_list())

        # Choose random numbers deterministically so we apply the same transforms
        # every time we load an image.
        rng = np.random.default_rng(self.seed + i)
        def coin_flip():
            return bool(rng.integers(2))

        # Flip
        if coin_flip():
            pattern = np.flip(pattern, axis=0)
        if coin_flip():
            pattern = np.flip(pattern, axis=1)

        # Rotate
        pattern = np.rot90(pattern, k=rng.integers(4))

        # Place the pattern into the initial state at a random position.
        initial_state = np.zeros((WORLD_SIZE, WORLD_SIZE))
        initial_state[:pattern.shape[0], :pattern.shape[1]] = pattern
        initial_state = np.roll(initial_state, rng.integers(WORLD_SIZE), axis=0)
        initial_state = np.roll(initial_state, rng.integers(WORLD_SIZE), axis=1)

        # Torch needs the pattern in a contiguous block of memory.
        initial_state = np.ascontiguousarray(initial_state)

        # Step simulation
        period = pattern_data.select('period').item()
        if period > 1:
            steps = rng.integers(period)
            if steps > 0:
                # Simulate for some number of steps, then grab the last frame.
                initial_state = simulate_one(initial_state, steps)[-1]

        return initial_state


    def __getitem__(self, i):
        # Figure out which pattern this dataset index corresponds to. We do
        # this by finding the row with the smallest index_threshold that's
        # greater than the target index. This respects the over sampling
        # specified using the multiplier column.
        pattern_data = self.df.filter(
            i < pl.col('index_threshold')
        ).filter(
            pl.col('index_threshold') == pl.col('index_threshold').min()
        )

        # Return a sample from the Dataset for training.
        return {
            # Generate some randomized initial state from this pattern.
            # NOTE: To work with video data, use a DataLoader to fetch a whole
            # batch of initial states, then simulate them in one shot before
            # passing them in as training data.
            'initial_state': self.random_initial_state(i, pattern_data),
            # What category of pattern is this (still_life, oscillator,
            # spaceship, or other)?
            'category': pattern_data.select('category').to_numpy().flatten(),
            # What is the period of this pattern?
            'period': pattern_data.select('period').to_numpy().flatten(),
            # Which pattern is this (one of the top 15 or 'other')
            'pattern_id': pattern_data.select('top_15').to_numpy().flatten(),
        }


def load_filtered_data():
    return load_catagolue_data().filter(
        # Filter out any patterns too big to fit in our simulation
        (pl.col('width') < WORLD_SIZE) &
        (pl.col('height') < WORLD_SIZE) &
        # Filter out any patterns with a period bigger than we can detect
        (pl.col('period') < MAX_PERIOD)
    )


def split_df(df):
    # Randomly shuffle a list of row indices into this data frame.
    size = len(df)
    all_rows = np.arange(size)
    np.random.shuffle(all_rows)

    # Split the shuffled indices up into groups sized to according to our
    # SPLIT_RATIO
    sections = []
    for frac in (val / sum(SPLIT_RATIO) for val in SPLIT_RATIO):
        prev = sections[-1] if sections else 0
        sections.append(int(prev + frac * size))
    row_groups = np.array_split(all_rows, sections)[:3]
    return [df[row_group] for row_group in row_groups]


def get_top_15_datasets():
    df = load_filtered_data()
    other = TOP_15_NAMES.index('other')

    # Create a dataframe containing only patterns not in the top 15.
    # TODO: Represent categories evenly.
    others = df.filter(
        pl.col('top_15') == other
    ).sample(
        # Keep just a random sampling of the dataset to reach the target size.
        # Half of the data will come from the top 15.
        DATASET_SIZE // VARIATIONS_PER_IMAGE // 2
    ).with_columns(
        # Every pattern not in the top 15 is represented once.
        multiplier=1
    )

    # And another with just the top 15.
    top_15 = df.filter(
        pl.col('top_15') != other
    )

    # Pick a random sampling of non-top-15 patterns for the three splits.
    train_other, validate_other, test_other = split_df(others)

    # Add the top 15 patterns to all three of the splits. Note that we will
    # randomly generate initial states from these patterns, so it's unlikely /
    # rare that the same initial_state will appear in both train and test,
    # though some of the random variations will be quite similar. Note that in
    # this case we actually want our model to memorize the top_15 patterns and
    # not generalize, so this should be okay.
    train_all = pl.concat([
        train_other,
        # The top 15 get oversampled so in total they have same representation
        # as the other category.
        top_15.with_columns(
            multiplier=len(train_other) // 15
        )
    ])
    validate_all = pl.concat([
        validate_other,
        # The top 15 get oversampled so in total they have same representation
        # as the other category.
        top_15.with_columns(
            multiplier=len(validate_other) // 15
        )
    ])
    test_all = pl.concat([
        test_other,
        # The top 15 get oversampled so in total they have same representation
        # as the other category.
        top_15.with_columns(
            multiplier=len(test_other) // 15
        )
    ])

    # Build DataSets from the split data. Make sure all three have a unique
    # seed so that the patterns that appear in all three splits will be
    # randomized differently and not repeat.
    return (
        GameOfLifeDataset(train_all, seed=0),
        GameOfLifeDataset(validate_all, seed=len(train_all)),
        GameOfLifeDataset(test_all, seed=len(train_all)+len(validate_all))
    )


def get_category_datasets():
    df = load_filtered_data()

    categories = df['category'].unique().to_list()
    target_size = DATASET_SIZE // VARIATIONS_PER_IMAGE // len(categories)

    # Do a train / validate / test split on all the data.
    train_all = pl.DataFrame()
    validate_all = pl.DataFrame()
    test_all = pl.DataFrame()
    # For each category
    for category in categories:
        # Isolate that category's pattern data.
        category_data = df.filter(
            pl.col('category') == category
        )
        category_size = len(category_data)
        # If this category is bigger than the target size, downsample.
        if category_size > target_size:
            category_data = category_data.sample(
                target_size
            ).with_columns(
                multiplier=1
            )
        # If this category is smaller than the target size, upsample.
        else:
            category_data = category_data.with_columns(
                multiplier = target_size // category_size
            )

        # Split the patterns from each category such that each pattern will
        # only appear in one of train, validate, or test.
        train_cat, validate_cat, test_cat = split_df(category_data)

        # Merge the per-category data.
        train_all = pl.concat((train_all, train_cat))
        validate_all = pl.concat((validate_all, validate_cat))
        test_all = pl.concat((test_all, test_cat))

    # Build DataSets from the split data.
    return (
        GameOfLifeDataset(train_all),
        GameOfLifeDataset(validate_all),
        GameOfLifeDataset(test_all)
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_data, validate_data, test_data = get_top_15_datasets()
    dataloader = DataLoader(train_data, batch_size=25, shuffle=True)
    for batch in dataloader:
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(batch['initial_state'][i].squeeze(), cmap='Greys')
            category = CATEGORY_NAMES[batch['category'][i]]
            period = batch['period'][i].item()
            name = TOP_15_NAMES[batch['pattern_id'][i]]
            ax = plt.gca()
            ax.set_title(f'{name}, {category}, {period}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.show()
