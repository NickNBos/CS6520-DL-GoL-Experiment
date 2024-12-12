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
        
        
        # Perhaps once the col index has outpaced the boundary, reset it to zero
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


def truncate_labels(data, labels, pad_size = 1):
    data = np.array(data)
    labels = np.array(labels)
    
    # There's an easy enough way to parallelize this, but I'd rather not
    for label in labels:
        # The maximum bounds are the label shape (really the world shape)
        x, y = label.shape
        
        for i in range(x):
            x_lower = max(0, i-pad_size)
            x_upper = min(i+pad_size, x) + 1
            for j in range(y):
                y_lower = max(0, j-pad_size)
                y_upper = min(j+pad_size, y) + 1
                
                # Erase the label where the actual pattern's live cells
                # are too far away
                label[i,j] = label[i, j] * np.max(data[x_lower:x_upper,y_lower:y_upper])
    
    return labels.tolist()
    
    
def visualize_world(data, label):
    # plt.figure('Data', clear=True)
    # plt.imshow(data, cmap="Greys")
    # plt.figure('Labels', clear=True)
    
    label = np.array(label)
    # colors = ['Reds','Greens','Blues', 'Purples']
    combined_label = np.zeros(label.shape[1:])
    for idx, channel in enumerate(label):
        # color = colors[idx]
        combined_label += 2*(idx+1)*np.array(channel)
        # plt.subplot(2,2,idx+1)
        # plt.imshow(channel, cmap=color, alpha = 0.3)
    
    plt.figure('combo', clear=True)
    plt.imshow(combined_label, cmap='YlGnBu', alpha = 0.9)
    plt.imshow(data, cmap="Greys", alpha=0.6)
    
    
    
def generate_many(world_count = 1000):
    if pattern_df is None:
        print('No catalogue of patterns to build from.')
        return None
    
    worlds = []
    labels = []
    truncated_labels = []
    for i in range(world_count):
        print(f"{i}/{world_count}")
        
        data, label = generate_one()
        
        t_l = truncate_labels(data, label)
        
        worlds.append(data)
        labels.append(label)
        truncated_labels.append(t_l)
    
    world_df = pl.DataFrame({"world pattern":worlds,
                             "label":labels,
                             "tight label":truncated_labels})
    
    if OUTPUT_PATH.exists():
        old_df = pl.read_parquet(OUTPUT_PATH)
        old_df.extend(world_df)
        old_df.write_parquet(OUTPUT_PATH)
    else:
        world_df.write_parquet(OUTPUT_PATH)
    
    
if __name__ == '__main__':
    # generate_many(1000)
    world_df = load_world_df()
    
    # Fix first 1000
    
    # for idx in range(1000):
    #     row = world_df[idx]
    #     world_df[idx, 'tight_labels'] = truncate_labels(row['world pattern'][0].to_list(), row['label'][0].to_list())
    
    
    
    # print(world_df)
    # world_instance = world_df[np.random.randint(len(world_df))]
    tightened_labels = []
    for world_idx in range(1000):
        print(world_idx)
        world_instance = world_df[world_idx]
        d = world_instance['world pattern'][0].to_list()
        l = world_instance['label'][0].to_list()
        t_l = truncate_labels(d, l)
        tightened_labels.append(t_l)
    
    for world_idx in range(1000,len(world_df)):
        print(world_idx)
        world_instance = world_df[world_idx]
        t_l = world_instance['tight_labels']
        tightened_labels.append(t_l[0].to_list())
    
    
    s = pl.Series('tight label', tightened_labels)
    world_df.replace_column(2, s)
    print(world_df)
        
    # d = world_instance['world pattern'][0].to_list()
    # t = world_instance['tight label'][0].to_list()
    # world_df = world_df.with_columns(
    #     tight_labels = tightened_labels
    # )
    # world_df.write_parquet(OUTPUT_PATH)
    # print(world_df)
    # l = truncate_labels(d, l,2)
    
    # visualize_world(d, t)
    
    ex = world_df[5000]
    
    d = ex['world pattern'][0].to_list()
    t = ex['tight label'][0].to_list()
    visualize_world(d, t)