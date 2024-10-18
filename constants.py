# The simulated Game of Life space will be a square with edges of this length.
WORLD_SIZE = 32

# The longest period that we expect this system to detect.
MAX_PERIOD = 16

# How long each simulation video to analyze will be.
VIDEO_LEN = MAX_PERIOD * 3

# How many augmented variations of each image to put in the dataset.
VARIATIONS_PER_IMAGE = 10

# The desired size of the full dataset, before the train / validate / test
# split. The actual size may be slightly smaller than this.
# NOTE: The smallest size that works with the current implementation of data
# set and our database is 11,000.
DATASET_SIZE = 11_000

# Training parameters
BATCH_SIZE = 250
NUM_EPOCHS = 10
