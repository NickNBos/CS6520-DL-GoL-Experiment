import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GameOfLife(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode='circular', # Wrap around / toroidal space
            bias=True,
        ).to(DEVICE)

        # The Game of Life ruleset encoded into CNN weights and bias.
        self.step.weight.data = torch.tensor([
            0.5, 0.5, 0.5,
            0.5, 0.2, 0.5,
            0.5, 0.5, 0.5
        ]).view(self.step.weight.shape).to(DEVICE)
        self.step.bias.data = torch.tensor([-1.0]).to(DEVICE)

    def forward(self, x):
        # Convolve to apply the Game of Life ruleset to every cell.
        x = self.step(x)
        # A value between 0.0 and 1.0 indicates this cell should be alive (1.0)
        return 1.0 * ((x > 0.0) & (x < 1.0))


simulator = torch.compile(GameOfLife())

def simulate_batch(initial_states, steps):
    batch_size, height, width = initial_states.shape
    num_channels = 1

    frames = torch.zeros((batch_size, steps + 1, num_channels, height, width))
    frames[:, 0, 0] = torch.from_numpy(initial_states)
    frames = frames.to(DEVICE)

    with torch.no_grad():
        for step in range(1, steps + 1):
            frames[:, step] = simulator(frames[:, step - 1])

    # Drop the channel dimension, since there's always one channel.
    return frames.cpu().numpy()[:, :, 0, :, :]


def simulate_one(initial_state, steps):
    return simulate_batch(np.expand_dims(initial_state, 0), steps).squeeze()


# A demo to simulate and display a glider to show this is working.
if __name__ == '__main__':
    import polars as pl
    from visualize import view_animation
    from import_data import load_catagolue_data

    # Load and filter pattern data.
    df = load_catagolue_data().filter(
        pl.col('category') == 3 # fizzlers
    )

    for _ in range(5):
        pattern_data = df.filter(
            (pl.col('width') < 32) & (pl.col('height') < 32)
        ).sample()
        for column in pattern_data.columns:
            print(column, pattern_data[column].item())
        pattern = np.array(pattern_data['pattern'].to_list()).squeeze()

        # Animate a simple scene containing this pattern.
        initial_state = np.zeros((32, 32))
        initial_state[:pattern.shape[0], :pattern.shape[1]] = pattern
        frames = simulate_one(initial_state, 50)
        view_animation(frames)
