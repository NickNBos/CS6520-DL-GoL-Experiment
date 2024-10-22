'''
The goal here is to generate some number of worlds to put into a dataset
For the training of the segmenter

The structure of such a dataset is as follows:
    Col 1: The data
        This will be the 'world' (of size (LARGE_WORLD_SIZE,LARGE_WORLD_SIZE)) containing some arbitrary number of non-colliding patterns
        Each pattern should be taken from the original 'catalogue.parquet' dataset
        Each pattern can be rotated (<= 3 times) and flipped (<= once) 
    Col 2: The labels
        This will be the mapping of the pattern-types to the world space, with size (category_count, LARGE_WORLD_SIZE, LARGE_WORLD_SIZE)
            The third(?) dimension, of size category_count, will map the patterns to the appropriate type
            by having each channel in the dimension corresponding to a category (fizzler, still-life, oscillator, spaceship)
            
Each column should contain multi-dimensional lists of purely 0's and 1's
'''
import constants

from urllib import request
from pathlib import Path

import numpy as np
import polars as pl

LARGE_WORLD_SIZE = constants.WORLD_SIZE * 2

OUTPUT_PATH = Path('world_catagolue.parquet')

INPUT_PATH = Path('catagolue.parquet')


CATEGORY_COUNT = 4

def load_catagolue_data():
    if OUTPUT_PATH.exists():
        return pl.read_parquet(OUTPUT_PATH)
    else:
        return None

pattern_df = load_catagolue_data()

    
def generate_one():
    data = np.zeros([LARGE_WORLD_SIZE,LARGE_WORLD_SIZE])
    label = np.zeros([CATEGORY_COUNT, LARGE_WORLD_SIZE,LARGE_WORLD_SIZE])
    
    # Start at 0, then move to not clash
    # Perhaps try to remember the fartherst you've reached in columns or rows
    # Then move some random length from there (np.random.randint(3) or so)
    col_index = 0
    row_index = 0
    
    while col_index < LARGE_WORLD_SIZE or row_index < LARGE_WORLD_SIZE:
        # Find a pattern
        # May need to seperate out the categories to get a useful distribution,
        # instead of purely still-lives
        # Use:
            # pattern_df.filter(pl.col("category") == CATEGORY_ID)
        # to parse the df into different categories, if needed
        
        
        pattern = pattern_df[np.random.randint(len(pattern_df))]
        # The dict indexing gives a series, take the first item from that series!
        pattern_category = pattern['category'][0] # This gives an int
        pattern_pattern = np.array(pattern['pattern'][0].to_list()) # This needed fixing to get the right shape

        # modify it (flip, rotate)
        # Maybe flip (0 or 1)
        flip = np.random.randint(2)
        if flip == 1:
            # TODO: do the flip
            pass
        
        # Rotate <=3 times (0,1,2,3)
        rotate = np.random.randint(4)
        while rotate > 0:
            # TODO: Do the rotate
            rotate -= 1
            
            
        # TODO: and attempt to put it in
        # DO NOT put it in if it would escape the boundaries
        # HOWEVER still increment col or row index
        
        # Pehaps once the col index has outpaced the boundary, reset it to zero
        # and shift the row index up past the maximum row observed
        
        pass
    
    
    return data.tolist(), label.tolist()
    

def generate_many(world_count = 1000):
    if pattern_df is None:
        print('No catalogue of patterns to build from.')
        return None
    
    worlds = []
    labels = []
    
    for _ in range(world_count):
        data, label = generate_one()
        
        worlds.append(data)
        labels.append(label)
    
    
    world_df = pl.DataFrame({"world pattern":worlds,
                             "label":labels})
    
    world_df.write_parquet(OUTPUT_PATH)
    
    
if __name__ == '__main__':
    generate_many(1)
