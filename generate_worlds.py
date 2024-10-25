'''
The goal here is to generate some number of worlds to put into a dataset
For the training of the segmenter

Note that I clearly have no idea which is the row and which is the column,
however my stunning consistency means it doesn't really matter.


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

from pathlib import Path

import numpy as np
import polars as pl

from matplotlib import pyplot as plt

LARGE_WORLD_SIZE = constants.WORLD_SIZE * 2

OUTPUT_PATH = Path('world_catagolue.parquet')

INPUT_PATH = Path('catagolue.parquet')


CATEGORY_COUNT = 4

def load_catagolue_data():
    if INPUT_PATH.exists():
        return pl.read_parquet(INPUT_PATH)
    else:
        return None

pattern_df = load_catagolue_data()


def load_world_df():
    if OUTPUT_PATH.exists():
        return pl.read_parquet(OUTPUT_PATH)
    else:
        print('No worlds file')
        return None
    
def generate_one():
    data = np.zeros([LARGE_WORLD_SIZE,LARGE_WORLD_SIZE])
    label = np.zeros([CATEGORY_COUNT, LARGE_WORLD_SIZE,LARGE_WORLD_SIZE])
    
    # May need to seperate out the categories to get a useful distribution,
    # instead of purely still-lives
    # Use:
        # pattern_df.filter(pl.col("category") == CATEGORY_ID)
    # to parse the df into different categories, if needed
    
    col_index = 0
    row_index = 0
    max_height_seen = 0
    
    drawn_from_df = pattern_df
    
    while col_index < LARGE_WORLD_SIZE or row_index < LARGE_WORLD_SIZE:
        # Find a pattern
        to_draw = np.random.randint(4)
        
        # To simplify, give an equal chance of finding any category
        drawn_from_df = pattern_df.filter(pl.col("category") == to_draw)
        
        pattern = drawn_from_df[np.random.randint(len(drawn_from_df))]
        # The dict indexing gives a series, take the first item from that series!
        pattern_category = pattern['category'][0] # This gives an int
        pattern_pattern = np.array(pattern['pattern'][0].to_list()) # This needed fixing to get the right shape
        
        # modify it (flip, rotate)
        # Maybe flip (0 or 1)
        flip = np.random.randint(2)
        if flip == 1:
            pattern_pattern = np.flip(pattern_pattern)
        
        # Rotate <=3 times (0,1,2,3)
        rotations = np.random.randint(4)
        if rotations > 0:
            pattern_pattern = np.rot90(pattern_pattern, rotations)
            
        width, height = pattern_pattern.shape
        # DO NOT put it in if it would escape the boundaries
        # HOWEVER still increment col or row index
        col_offset = col_index + width
        row_offset = row_index + height
        
        if col_offset < LARGE_WORLD_SIZE:
            if row_offset < LARGE_WORLD_SIZE:
                # It will fit
                data[col_index:col_offset, row_index:row_offset] = pattern_pattern
                
                label[pattern_category][col_index:col_offset, row_index:row_offset] = np.ones([width, height],dtype=int)
                # Record the highest height seen
                if row_offset > max_height_seen:
                    max_height_seen = row_offset
                
                col_index = col_offset
        
        
        # Pehaps once the col index has outpaced the boundary, reset it to zero
        # and shift the row index up past the maximum row observed
        
        # Move at least 1, at most 4
        # Do this even if the placement failed
        col_index += np.random.randint(4) + 1
        
        if col_index > LARGE_WORLD_SIZE:
            max_height_seen += np.random.randint(4) + 1
            row_index = max_height_seen
            if row_index < LARGE_WORLD_SIZE:
                col_index = 0
    
    return data.tolist(), label.tolist()


def visualize_world(data, label):
    plt.figure('Data', clear=True)
    plt.imshow(data, cmap="Greys")
    plt.figure('Labels', clear=True)
    for idx, channel in enumerate(label):
        plt.subplot(2,2,idx+1)
        plt.imshow(channel, cmap="Greys")
    

def generate_many(world_count = 1000):
    if pattern_df is None:
        print('No catalogue of patterns to build from.')
        return None
    
    worlds = []
    labels = []
    
    for i in range(world_count):
        print(f"{i}/{world_count}")
        
        data, label = generate_one()
        
        worlds.append(data)
        labels.append(label)
    
    
    world_df = pl.DataFrame({"world pattern":worlds,
                             "label":labels})
    
    world_df.write_parquet(OUTPUT_PATH)
    
    
if __name__ == '__main__':
    # generate_many()
    world_df = load_world_df()
    
    world_instance = world_df[np.random.randint(len(world_df))]
    
    visualize_world(world_instance['world pattern'][0].to_list(), world_instance['label'][0].to_list())