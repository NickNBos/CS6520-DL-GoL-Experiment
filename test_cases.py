import re

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import torch

from constants import WORLD_SIZE
from dataset import load_filtered_data
import image_classifier, video_classifier
from simulate import simulate_batch
from metrics import MetricsTracker
from import_data import CATEGORY_NAMES, TOP_15_NAMES

catagolue_df = load_filtered_data()


# Taken from https://conwaylife.com/forums/viewtopic.php?f=4&t=4835
predecessor_dict = {
    'block': ['2b2o$bo$o', '2bo$bobo$o', '2o$o', 'b2o$o2bo'],
    'blinker': ['bo$obo2$bo', 'bo$obo2$o', 'o$b2o2$o', 'o$obo2$2bo', 'o$obo2$bo', 'o$obo2$o'],
    'beehive': ['2bo$2o2$bo', '2bo$2o2$o', '2bo$2ob2o', '4o', 'b2o$o2b2o', 'b3o2$o', 'bo$2o2$o', 'o$2o2$o', 'o$b2o$3bo'],
    'glider': ['2bo$b3o$o$o', '2o$2b2o$bo', '3o$3bo$o', '3o$o$2bo', '3o$o$3bo', 'bo$obo$3bo$2o', 'obo$bobo$bo'],
    'loaf': ['2b3o$bo$o', '2b3o$o$bo', '2bo$b3o$o', '2bo$b3o2$o', '2bo$obo2$2b2o', '3o$2bo$3bo', '3o$2bo$4bo', '3o$o$o', '3o2$3b2o', 'b2o$2o2$bo', 'b2o$2o2$o', 'b4o$o', 'bo$2obo2$2bo', 'bo$2obo2$o', 'bo$o$bo$2b2o', 'bo$o$bobo$2bo', 'bo$obo2$2b2o', 'bobo$2o2$o', 'o$3o2$3bo', 'o$b2o$2bo$3bo', 'o$b2o$3bo$4bo', 'o$o$bo$2b2o', 'o$obo2$2b2o', 'o2$b2o$2bo$3bo', 'o2$b2o$2bo$4bo', 'ob3o2$bo'],
    'boat': ['2b2o$bo$2o', '2b2o2$bo$2o', '2o$o$b2o', '3o$3bo$o3bo', 'bo$ob2o$o', 'bo$obo$bo$3bo', 'bo2$4o', 'bobo$obo$bo', 'bobo$obo$o', 'bobo2$bo$2o', 'o2$3o$3bo', 'o2$b2o$3b2o', 'o2$b2obo$3bo'],
    'ship': ['2b3o$bo$o$o$o', '2o$bo$2b2o2$4bo', '2o$o$b2o2$3bo', '2o$o2bo$2bo$bo', '2o$obo$bo$3bo', 'b2o$o2bo$o$bo', 'bo$3bo$obo$2o', 'bo$bo$o$b2o2$3bo', 'bo$ob2o$bo$bo', 'o$2b2o$b2o$bo', 'o$2bo$bobo$2bo$4bo', 'o$2bo$bobo$2bobo', 'o$bo$o$b2o2$3bo', 'o$bo$obo$bobo', 'obo$bobo$2bobo'],
    'pond': ['bo$2o$2bo'],
    'tub': ['2o$2bo$2o$o', 'b2o$o$obo', 'bo$obo$bo', 'bo2$3o2$bo'],
    'long boat': ['2b3o$bo$o$bo', '2b3o$bo$o$o', 'bo2$2o2$2bo$2b2o', 'o2$2o2$2bo$2b2o'],
    'toad': ['2b2o2$bo$b2o2$o', '2bo2$obo$bobo2$bo', '3bo$obo$bobo$o', '3o$3bo2$b2o', '4bo$obo$bobo$o', '5bo$bobo$2bobo$o', 'bo$2b2o$bo$o$2bo', 'bo$2b2o$o$bo$2bo', 'bo$2o$2bo$3b2o', 'bo$2o$2bobo$3bo', 'bo$o2b2o$bo$2bo', 'bo$o2bo$bo$2bobo', 'bo$o2bo$bo2bo$2bo', 'bo2$2b2o$2o2$2bo', 'bo2$obo$bobo2$2bo', 'bo2$obo$bobo2$bo', 'bobo2$bo$b2o2$o', 'o$2bo$bo$bo$2bo$3bo', 'o$2o$2b2o$4bo', 'o$2o$2b2o2$2bo', 'o$2o$2b2o2$3bo', 'o$3b2o$b2o$5bo', 'o$bo$2bo$bo$bo$3bo', 'o$bobo$2bobo$5bo', 'o2$b2o$b2o2$3bo', 'o2$obo$bobo2$2bo', 'o2$obo$bobo2$3bo', 'o2$obo$bobo2$bo', 'obo2$bo$b2o$3bo', 'obo2$bo$b2o2$3bo'],
    'ship-tie': ['4b2o$4bo$2b2o2$bo$2o', 'bo$bo$o$b2o$3bo$3bob2o$4bo'],
    'beacon': ['3b2o$5bo$3b2o$obo$obo$bo', 'bo$2o$2b2o$2bo', 'bo$o$bo$2bobo$3bo', 'bo2$2o$2bobo$2bo', 'o$bo$bo$2b2o$4bo', 'o$o$bo$2bo$3b2o', 'o$o$bo$2bobo$3bo', 'o2$2o$2bo$2bobo', 'o2$2o$2bobo$2bo'],
    'barge': ['2o$2bo$bo$2b2o', '3o$3bo$4b2o', '3o$3bobo$4bo', 'o$o$b2o$2bo$3b2o', 'o2$b2o$3b2o2$5bo'],
    'half-bakery': ['o$bo$bo$2bo$3bo$4b2o$6bo', 'o$o$o$bo$2bo$3b3o']
}


def expand_rle(rle):
    number = re.compile('\\d+')
    expanded = ''
    i = 0
    # Look for every number embedded in the string...
    while match := number.search(rle, i):
        start, end = match.span()

        # Add any non-number characters between the last match and this one to
        # the expanded string, since they aren't being repeated.
        if start > i:
            expanded += rle[i:start]

        # Replace the number with the character that appears after it, repeated
        # the specified number of times.
        repeats = int(match.group(0))
        expanded += rle[end] * repeats

        # Move our index to the point after the repeated character.
        i = end + 1

    # Add any non-repeating characters from after the final match to the
    # expanded string.
    expanded += rle[i:]
    return expanded


def decode_rle(rle):
    # Expand the rle string, break it up into lines, and allocate an array to
    # hold the decoded pattern.
    lines = expand_rle(rle).split('$')
    rows = len(lines)
    cols = max(map(len, lines))
    result = np.zeros((rows, cols))

    # Go through every character to set the live cells in this pattern.
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == 'o':
                result[r, c] = 1
    return result


def make_worlds(patterns):
    result = np.zeros((len(patterns), 1, WORLD_SIZE, WORLD_SIZE))
    for i, pattern in enumerate(patterns):
        rows, cols = pattern.shape
        result[i, 0, :rows, :cols] = pattern
    return result


def get_predecessors():
    all_patterns = []
    all_categories = []
    all_pattern_ids = []
    # Go through all the RLE patterns in the dict above and turn them into
    # tensors similar to what you'd get from the real dataset.
    for (name, rle_list) in predecessor_dict.items():
        patterns = make_worlds([decode_rle(rle) for rle in rle_list])
        pattern_id = TOP_15_NAMES.index(name)
        category = catagolue_df.filter(
            pl.col('top_15') == pattern_id
        )['category'].item()
        all_patterns.extend(patterns)
        all_categories.extend([category] * len(patterns))
        all_pattern_ids.extend([pattern_id] * len(patterns))
    return (torch.tensor(np.array(all_patterns)).to(torch.float).cuda(),
            torch.tensor(all_categories).cuda(),
            torch.tensor(all_pattern_ids).cuda())


def get_models():
    return {
        'image_classifier': image_classifier.get_model().cuda(),
        'video_classifier': video_classifier.get_model().cuda()
    }


def test_predecessors():
    initial_state, true_category, true_pattern_id = get_predecessors()
    for name, model in get_models().items():
        metrics_tracker = MetricsTracker()
        pred_category_oh, pred_pattern_id_oh = model.forward(initial_state)

        print(f'Results for {name}')
        metrics_tracker.score_batch(
            true_category, pred_category_oh,
            true_pattern_id, pred_pattern_id_oh, mode='test')
        with open(f'output/{name}/test_predecessor.txt', 'w') as file:
            metrics_tracker.print_summary(mode='test', file=file)


def get_variants(patterns):
    num_patterns = len(patterns)
    num_variations = 50

    # Prepare to simulate many variations of all patterns.
    initial_states = torch.zeros(
        (num_patterns, num_variations, WORLD_SIZE, WORLD_SIZE))
    for p, pattern in enumerate(patterns):
        for v in range(num_variations):
            # Generate a world-sized initial state from this pattern.
            rows, cols = pattern.shape
            initial_states[p, v, 1:rows+1, 1:cols+1] = pattern

            # Keep one variant unchanged. For all the others, randomly change
            # just one cell.
            if v > 0:
                row = np.random.randint(0, rows+2) % WORLD_SIZE
                col = np.random.randint(0, cols+2) % WORLD_SIZE
                initial_states[p, v, row, col] = 1 - initial_states[p, v, row, col]

    # Simulate all the variants of all the patterns and grab the last frame.
    final_state = simulate_batch(
        initial_states.reshape(-1, 1, WORLD_SIZE, WORLD_SIZE), 50
    )[:, -1, 0, :, :]

    # Separate out the final state from the unmodified variant from all the
    # modified ones.
    expected = final_state[::num_variations]
    variants = final_state.reshape(
        -1, num_variations, WORLD_SIZE, WORLD_SIZE
    )[:, 1:, :, :].reshape(num_patterns, -1, WORLD_SIZE, WORLD_SIZE)

    # Go through all the patterns and find examples where changing one pixel
    # did not affect the final results, and where it resulted in the pattern
    # completely dying out (a "fizzler").
    unchanged = []
    fizzlers = []
    for p in range(num_patterns):
        sample_unchanged = None
        sample_fizzler = None

        # Go through all the variants of this pattern and save the initial
        # state from one example of each type.
        for v, variant in enumerate(variants[p]):
            if torch.equal(expected[p], variant):
                sample_unchanged = initial_states[p, v + 1]
            if torch.count_nonzero(variant) == 0:
                sample_fizzler = initial_states[p, v + 1]
        unchanged.append(sample_unchanged)
        fizzlers.append(sample_fizzler)

    # Returns two lists parallel with the input patterns. Each list contains
    # one example of a one-cell that has no effect or makes the pattern fizzle,
    # if one could be found. Nones appear in the list where no such example
    # could be found.
    return unchanged, fizzlers


def test_variants():
    # Pick the top 15 patterns plus 15 other examples from each category.
    df = pl.concat([
        catagolue_df.filter(pl.col('top_15') < 15),
        catagolue_df.filter(
            (pl.col('top_15') == 15) & (pl.col('category') == 0)).sample(15),
        catagolue_df.filter(
            (pl.col('top_15') == 15) & (pl.col('category') == 1)).sample(15),
        catagolue_df.filter(
            (pl.col('top_15') == 15) & (pl.col('category') == 2)).sample(15),
    ])

    # Try randomly tweaking each pattern by one cell and find examples where
    # that either had no effect, or made the pattern die out completely.
    patterns = [torch.tensor(pattern) for pattern in df['pattern']]
    unchanged, fizzlers = get_variants(patterns)

    # Merge all the examples we found into flat lists for checking against our
    # model.
    initial_state = []
    true_category = []
    true_pattern_id = []
    for (c, t, u, f) in zip(df['category'], df['top_15'], unchanged, fizzlers):
        if u is not None:
            initial_state.append(u)
            true_category.append(c)
            true_pattern_id.append(t)
        if f is not None:
            initial_state.append(f)
            true_category.append(CATEGORY_NAMES.index('other'))
            true_pattern_id.append(TOP_15_NAMES.index('other'))

    # Put all the data into cuda tensors of the right shape.
    initial_state = torch.stack(
        initial_state
    ).reshape(-1, 1, WORLD_SIZE, WORLD_SIZE).cuda()
    true_category = torch.tensor(true_category).cuda()
    true_pattern_id = torch.tensor(true_pattern_id).cuda()

    # Now run both models on these variations and see how they perform.
    for name, model in get_models().items():
        metrics_tracker = MetricsTracker()
        pred_category_oh, pred_pattern_id_oh = model.forward(initial_state)

        print(f'Results for {name}')
        metrics_tracker.score_batch(
            true_category, pred_category_oh,
            true_pattern_id, pred_pattern_id_oh, mode='test')
        with open(f'output/{name}/test_variants.txt', 'w') as file:
            metrics_tracker.print_summary(mode='test', file=file)


if __name__ == '__main__':
    print('Testing on predecessors of the top 15 patterns...')
    test_predecessors()

    print('Testing on patterns modified by one cell...')
    test_variants()
