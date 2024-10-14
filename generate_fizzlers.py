import numpy as np

# For some reason, I need this to let it run (likely setup torch weird)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import simulate

# import polars as pl
# from pathlib import Path
# import import_data


# OUTPUT_PATH = Path('catagolue.parquet')
# df = pl.read_parquet(OUTPUT_PATH)

def make_pattern(size):
    # Make a pattern of shape up to size x size
    potential_pattern = np.random.randint(0,2,(size,size))
    
    # It is NOT viable to try to get every combination, as that generates
    # 2**(size**2) combinations [Balloons QUICKLY]
    # Not really a problem, until you try to get size > 4
    
    # from itertools import product
    # elements = [0,1]
    # combinations = np.array(list(product(list(product(elements, repeat=size)), repeat=size)))

    # Add some arbitrary large padding to not let the pattern run into itself
    initial_states = np.zeros((64, 64))
    initial_states[3:3+size, 3:3+size] = potential_pattern
    
    frames = simulate.simulate_one(initial_states, 48)
    print(potential_pattern)
    print(initial_states)
    print(frames.shape, frames[-1].sum())
    
if __name__ == '__main__':
    make_pattern(5)
    