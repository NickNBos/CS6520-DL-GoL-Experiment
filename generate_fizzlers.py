import numpy as np

# For some reason, I need this to let it run (likely setup torch weird)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import simulate

from decode import Decoder

padder = Decoder()

def make_pattern(size = 3):
    # Make a pattern of shape up to size x size
    potential_pattern = np.random.randint(0,2,(size,size))
    
    # Keep generating if you get an all 0 array - those are no fun
    # (and would likely throw off the analyzer)
    while potential_pattern.sum() == 0:
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
    # print(potential_pattern)
    # print(initial_states)
    # print(frames.shape, frames[-1].sum())
    
    # Crunch out the x and y dims to retrieve frame sums    
    lifespan = max(np.where(np.sum(np.sum(frames, axis=1), axis=1) > 0)[0])
    
    if frames[-1].sum() == 0:
        # Add the standard padding to the pattern
        # Would be better to have this independent of the decoder, but it shouldn't matter
        return padder.standard_one_pad(potential_pattern).tolist(), lifespan
    else:
        return None, None
    
def generate(size = 5, goal = 500, tries = 5000):
    
    # Max out at some large number if tries is none
    if tries is None:
        tries = 65536
    
    pattern_list = []
    for _ in range(tries):
        pattern, lifespan = make_pattern(size)
        
        # Prevent duplicates
        if pattern is not None and [pattern, lifespan] not in pattern_list:
            pattern_list.append([pattern, lifespan])
            
            # Only need to check this if you've just added a pattern
            if len(pattern_list) >= goal:
                # Allow for an early return if the desired number of patterns is met
                return pattern_list
            
    return pattern_list


if __name__ == '__main__':
    fizzlers = generate(5)
    