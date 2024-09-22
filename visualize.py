import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCALE_FACTOR = 4
MS_PER_FRAME = 100
BORDER_COLOR = '#007155'  # UVM Green


def save_animation(frames, path):
    def process(frame):
        # Scale up the image to make it easier to see.
        frame = frame.repeat(SCALE_FACTOR, 0).repeat(SCALE_FACTOR, 1)
        # Convert from 1.0 / 0.0 to black / white.
        return 0xFF - 0xFF * frame.astype(np.uint8)

    images = [
        Image.fromarray(process(frame), mode='L')
        for frame in frames
    ]
    durations = [MS_PER_FRAME] * len(frames)
    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0,
        duration=durations)


def view_animation(frames):
    fig = plt.figure()
    ax = plt.gca()

    # Formatting: show no grid, but draw a 1px border around the image.
    ax.grid(False)
    ax.spines[:].set_visible(True)
    plt.setp(ax.spines.values(), color=BORDER_COLOR)
    plt.setp(ax.spines.values(), linewidth=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # Start with showing frame 0
    image = ax.imshow(frames[0], cmap='gray_r', vmin=0.0, vmax=1.0)

    # Function to update the image being shown on each frame.
    def animate_func(i):
        image.set_array(frames[i])
        return image

    # The animation object must not be garbage collected until after calling
    # plt.show(), so keep a reference to it.
    anim = animation.FuncAnimation(
        fig, animate_func, frames=frames.shape[0], interval=MS_PER_FRAME)

    plt.show()
