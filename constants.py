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
DATASET_SIZE = 500_000
