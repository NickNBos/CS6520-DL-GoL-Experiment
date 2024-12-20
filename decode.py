import numpy as np
import matplotlib.pyplot as plt

from constants import WORLD_SIZE

class Decoder:
    def __init__(self):
        self.bitstring_map = {}
        self.y_map = {}
    
        # First, set up the map for characters to bitstrings
        # Get 0-9 in there
        item_list = [str(num) for num in range(10)]
    
        # Get a-v in there
        item_list.extend([chr(i) for i in range(ord('a'), ord('v') + 1)])
    
        for index, item in enumerate(item_list):
            # Cutoff the '0b'
            bitstring_rep = bin(index)[2:]
    
            # Make sure the representation is exactly 5 bits
            while len(bitstring_rep) < 5:
                bitstring_rep = '0' + bitstring_rep
    
            # Store the representation to each item 0-9 a-v
            self.bitstring_map[item] = bitstring_rep
    
        # Then, set up the map for y codes to 0's
    
        # Get the rest of the letters in there
        item_list.extend([chr(i) for i in range(ord('w'), ord('z') + 1)])
    
        # deja-vu
        for index, item in enumerate(item_list):
            # explicitly list out y0, y1, ..., yy, yz
            y_code = 'y' + item
    
            # y0 = 0000, y1 = 00000, ... (so index plus four)
            zero_string = '0' * (index + 4)
    
            self.y_map[y_code] = zero_string
    
    
    def decode(self, x_string):
        # Ignore everything before the underscore
        x_string = x_string.split('_')[1]
        # First, process all non - 0-9 a-v characters first
        # So w, x, y, z
    
        # MUST start with the 'y' codes, as those are two characters
        # Of these, start with yy, to avoid erroneously thinking that
        # the second y pairs with the following character
        x_string = x_string.replace('yy', self.y_map['yy'])
        
        for y_code, zero_string in self.y_map.items():
            x_string = x_string.replace(y_code, zero_string)
    
        # Then get w, x, and z
        x_string = x_string.replace('w', '00')
        x_string = x_string.replace('x', '000')
    
        x_strings = x_string.split('z')
    
    
        # Then process all the other characters to create the list
        list_rep = []
        for line in x_strings:
            line_rep = []
    
            for character in line:
                line_rep.append(self.bitstring_map[character])
    
            list_rep.append(line_rep)
    
    
        # Initialize the array representation
        rows = 5 * len(x_strings)
        columns = max([len(x_str) for x_str in x_strings])
    
        x_array = np.zeros((rows, columns), dtype=int)
    
    
        for line_num, line in enumerate(list_rep):
            base_row = 5 * line_num
            for column_num, column_info in enumerate(line):
                # At some point, it seems I reversed the columns meanings
                # so just reverse the columns again to fix
                x_array[base_row:base_row + 5, column_num] = [int(character) for character in reversed(column_info)]
    
        return x_array
    
    # Functions past this point expect the array form of x, from decode()
    def find_corners(self, x):
        # Retrieve the bottom left and top right corners
        locations = np.where(x==1)
        dim_1 = locations[0]
        dim_2 = locations[1]
        
        bottom_left = (min(dim_1), min(dim_2))
        top_right = (max(dim_1), max(dim_2))
        
        return (bottom_left, top_right)
    
    def clip_excess(self, x, corners):
        lows = corners[0]
        highs = corners[1]
        # Need to add 1, numpy has non-inclusive upper bounds
        return x[lows[0]:highs[0]+1, lows[1]:highs[1]+1]
    
    def add_padding(self, x, new_size = WORLD_SIZE):
        old_height, old_width = x.shape
    
        if type(new_size) is int:
            new_size = [new_size, new_size]
    
        assert old_height <= new_size[0], 'Height must be larger in new object'
        assert old_width <= new_size[1], 'Width must be larger in new object'
    
        new_x = np.zeros((new_size[0],new_size[1]), dtype=int)
    
        # Find a spot that will center the original image
        new_start_row = int(new_size[0]/2) - int(old_height/2)
        new_start_col = int(new_size[1]/2) - int(old_width/2)
    
        # Put the old data into the new, larger/uniform array
        new_x[new_start_row:new_start_row + old_height,
              new_start_col:new_start_col + old_width] = x
    
        return new_x
    
    def standard_one_pad(self, x):
        # Combo function for easy use in list comprehension
        clipped_x = self.clip_excess(x, self.find_corners(x))
        one_padding = np.array(clipped_x.shape) + 2
        return self.add_padding(clipped_x, one_padding)
    
    def visualize(self, x):
        plt.imshow(x, cmap="Greys")
    
        ax = plt.gca()
        ax.set_yticks(np.arange(-.5, x.shape[0]))
        ax.set_xticks(np.arange(-.5, x.shape[1]))
    
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
        ax.grid(linewidth=0.25)


if __name__ == '__main__':

    #example_list = ['xs4_33', 'xs4_252', 'xq4_a1hh197zx6777be4', 'xq4_027deee6z4eqscc6']
    example_list = ['xq4_i23m3y2ggz102778cew5soalzy8qqaammea6e0oggzya111y1115oc']
    decoder = Decoder()
    for example in example_list:
        decoded_example = decoder.decode(example)

        corners = decoder.find_corners(decoded_example)
        
        clipped_example = decoder.clip_excess(decoded_example, corners)
        
        slight_padding = np.array(clipped_example.shape) + 2
        
        resized_example = decoder.add_padding(clipped_example, slight_padding)
        
        print(resized_example)
        plt.figure(example, clear=True)
        decoder.visualize(decoder.standard_one_pad(clipped_example))
        # decoder.visualize(resized_example)
        plt.show()
        