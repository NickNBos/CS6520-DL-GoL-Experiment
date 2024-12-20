from pathlib import Path
import warnings

import polars as pl
import torch
from torch import nn, optim
from torchinfo import summary
from constants import MAX_PERIOD, VIDEO_LEN

from dataset import get_split_dataset
from image_classifier import CNNBlock
from import_data import CATEGORY_NAMES, TOP_15_NAMES
from metrics import MetricsTracker, compare_runs
from simulate import simulate_batch
from training import train_model, test_model

# Torch fires this warning on every call to load_state_dict()
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class Conv2Plus1d(nn.Module):
    def __init__(self, in_depth, out_depth):
        super(Conv2Plus1d, self).__init__()
        # Compute the interface size between the frame-wise and time-wise
        # convolutions so the number of parameters is the same as a simple 3D
        # convolution.
        hidden_size = (
            (3 * 3 * 3 * in_depth * out_depth) //
            (3 * 3 * in_depth + 3 * out_depth))

        # Here we replace a single 3D convolution with a 2D and 1D convolution,
        # design inspired by https://pytorch.org/vision/main/_modules/torchvision/models/video/resnet.html
        self.block = nn.Sequential(
            # First convolve spatial information in each frame using a 3x3 kernel
            nn.Conv3d(in_depth, hidden_size, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1), padding_mode='circular',
                      bias=False, stride=1),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(),

            # Then integrate temporal information across windows of three
            # contiuguous frames in the video using a 1x1 kernel.
            nn.Conv3d(hidden_size, out_depth, kernel_size=(3, 1, 1),
                      padding=(1, 0, 0), bias=False, stride=1),
            nn.BatchNorm3d(out_depth),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class VideoClassifier(nn.Module):
    def __init__(self, model_name='cnn', num_layers=1, video_len=VIDEO_LEN):
        super(VideoClassifier, self).__init__()
        self.video_len = video_len

        # A few different RNN variants to try, all with the same API.
        recurrent_layer_types = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }

        # TODO: Implement different temporal models...
        if model_name == 'cnn':
            frame_multiplier = (video_len + 1) // 2 // 2
            # This is a 2+1D variant of the "Minimal" network from image_classifier.py
            self.backbone = nn.Sequential(
                                     #     FxWxHxC -> FxWxHxC
                Conv2Plus1d(1, 32),  #  11x32x32x1 -> 11x32x32x32
                nn.AvgPool3d(2),     # 11x32x32x32 -> 5x16x16x32
                Conv2Plus1d(32, 64), #  5x16x16x32 -> 5x16x16x64
                nn.AvgPool3d(2),     #  5x16x16x64 -> 2x8x8x64
                nn.Flatten(),
                nn.Linear(frame_multiplier * 8 * 8 * 64, 1024),
                nn.ReLU()
            )
        elif model_name in recurrent_layer_types.keys():
            # This is just the "Minimal" network from image_classifier.py.
            # We'll do a forward pass of this on each frame of the video...
            self.backbone = nn.Sequential(
                CNNBlock(1, 32, 1),  # 32x32x1  -> 32x32x32
                nn.AvgPool2d(2),     # 32x32x32 -> 16x16x32
                CNNBlock(32, 64, 1), # 16x16x32 -> 16x16x64
                nn.AvgPool2d(2),     # 16x16x64 -> 8x8x64
                nn.Flatten(),
                nn.Linear(8 * 8 * 64, 1024),
                nn.ReLU()
            )
            # ... then we'll pass the features through the recurrent part of
            # the network for temporal integration.
            self.recurrent = recurrent_layer_types[model_name](
                1024, 1024, num_layers, batch_first=True)

        # Define two heads for the two classification tasks.
        self.category_head = nn.Sequential(
            nn.Linear(1024, len(CATEGORY_NAMES)))
        self.top_15_head = nn.Sequential(
            nn.Linear(1024, len(TOP_15_NAMES)))

    def forward(self, x):
        # Take a batch of initial states and simulate them all in parallel.
        x = simulate_batch(x, self.video_len)
        batch_size, steps, num_channels, height, width = x.shape

        if hasattr(self, 'recurrent'):
            # Pretend like each frame is just another sample in a larger batch, and
            # run the image classifier on each frame that way. Then, split it back
            # into a one set of 1024 features per frame per batch item.
            x = x.reshape(batch_size * steps, num_channels, height, width)
            features = self.backbone(x)
            # Now run the per-frame features through our recurrent network to
            # summarize the overall video.
            features = features.reshape(batch_size, steps, 1024)
            # Take just the final outputs, then pass them through one last ReLU
            # before classification.
            outputs, _ = self.recurrent(features)
            features = nn.functional.relu(outputs[:, -1, :])
        else:
            # Put the time dimension next to the spatial ones for 3D conv.
            x = torch.permute(x, (0, 2, 1, 3, 4))
            features = self.backbone(x)

        return self.category_head(features), self.top_15_head(features)


def hp_optimizer():
    path = Path('output/video_classifier/hp_optimizer')
    path.mkdir(exist_ok=True, parents=True)

    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adagrad': optim.Adagrad,
    }
    for optim_name, optimizer in optimizers.items():
        title = f'optimizer = {optim_name}'
        data_filename = path / f'train_log_{optim_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {optim_name}...')
            model = VideoClassifier()
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker, optimizer)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, optim_name, title)
    compare_runs(path, optimizers.keys())


def hp_model_arch():
    path = Path('output/video_classifier/hp_model_arch')
    path.mkdir(exist_ok=True, parents=True)

    models = {
        'cnn': {'model_name': 'cnn'},
        'gru1': {'model_name': 'gru', 'num_layers': 1},
        # 'gru2': {'model_name': 'gru', 'num_layers': 2},
        'rnn1': {'model_name': 'rnn', 'num_layers': 1},
        # 'rnn2': {'model_name': 'rnn', 'num_layers': 2},
        'lstm1': {'model_name': 'lstm', 'num_layers': 1},
        # 'lstm2': {'model_name': 'lstm', 'num_layers': 2},
    }
    for expt_name, expt_args in models.items():
        title = f'model_arch = {expt_name}'
        data_filename = path / f'train_log_{expt_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {expt_name}...')
            model = VideoClassifier(**expt_args)
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)
            with open(path / f'model_summary_{expt_name}.txt', 'w') as file:
                print(summary(model, verbose=0), file=file)

        metrics_tracker.summarize_training(path, expt_name, title)
    compare_runs(path, models.keys())


def hp_video_len():
    path = Path('output/video_classifier/hp_video_len')
    path.mkdir(exist_ok=True, parents=True)

    # Does the model need to see multiple cycles of a periodic pattern to pick
    # up on that information? Let's find out.
    cycle_options = [1, 2, 3]
    for num_cycles in cycle_options:
        video_len = MAX_PERIOD * num_cycles
        expt_name = f'{num_cycles}x{MAX_PERIOD}'
        expt_args = {'video_len': video_len}

        title = f'video_len = {video_len}'
        data_filename = path / f'train_log_{expt_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {expt_name}...')
            model = VideoClassifier(**expt_args)
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, expt_name, title)
    compare_runs(path, [f'{num_cycles}x{MAX_PERIOD}' for num_cycles in cycle_options])


def tune_hyperparameters():
    # print('Comparing different optimizers...')
    # hp_optimizer()

    print('Comparing different model architectures...')
    hp_model_arch()

    print('Comparing different video lengths...')
    hp_video_len()


def get_model():
    model = VideoClassifier()
    model.load_state_dict(
        torch.load('output/video_classifier/model.pt', weights_only=True))
    return model


if __name__ == '__main__':
    # tune_hyperparameters()

    path = Path('output/video_classifier')
    path.mkdir(exist_ok=True, parents=True)
    model_filename = path / 'model.pt'

    model = VideoClassifier()
    train_data, validate_data, test_data = get_split_dataset()
    metrics_tracker = MetricsTracker()

    if model_filename.exists():
        print('Using pre-trained model weights')
        model.load_state_dict(torch.load(model_filename, weights_only=True))
    else:
        print('Training model...')
        state_dict = train_model(model, train_data, validate_data, metrics_tracker)
        torch.save(state_dict, model_filename)
        model.load_state_dict(state_dict)
        metrics_tracker.get_summary().write_parquet(path / 'train_log.parquet')
        metrics_tracker.summarize_training(path)

    print('Evaluating on test dataset...')
    metrics_tracker.reset()
    test_model(model, test_data, metrics_tracker)
    metrics_tracker.print_summary('test')
