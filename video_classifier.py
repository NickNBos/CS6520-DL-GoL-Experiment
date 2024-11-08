from pathlib import Path
import warnings

import polars as pl
from torch import nn
from constants import MAX_PERIOD, VIDEO_LEN

from dataset import get_split_dataset
from image_classifier import CNNBlock
from import_data import CATEGORY_NAMES, TOP_15_NAMES
from metrics import MetricsTracker
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
                      padding=1, bias=False, stride=1),
            nn.BatchNorm3d(out_depth),
            nn.ReLU(),
        )

class VideoClassifier(nn.Module):
    def __init__(self, model_name='', num_layers=1, video_len=VIDEO_LEN):
        super(VideoClassifier, self).__init__()
        self.model_name = model_name
        self.video_len = video_len

        # A few different RNN variants to try, all with the same API.
        recurrent_layer_types = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }

        # TODO: Implement different temporal models...
        if model_name == 'cnn':
            # This is a 2+1D variant of the "Minimal" network from image_classifier.py
            self.backbone = nn.Sequential(
                Conv2Plus1d(1, 32),  # Vx32x32x1  -> Vx32x32x32
                nn.AvgPool3d(2),     # Vx32x32x32 -> Vx16x16x32
                Conv2Plus1d(32, 64), # Vx16x16x32 -> Vx16x16x64
                nn.AvgPool3d(2),     # Vx16x16x64 -> Vx8x8x64
                nn.Flatten(),
                nn.Linear(video_len * 8 * 8 * 64, 1024),
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
            self.recurrent = recurrent_layer_types['model_name'](
                1024, 1024, num_layers, batch_first=True)

        # Reuse the classification heads from the image model (they will get
        # retrained from scratch).
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
            _, features = self.recurrent(features)
        else:
            features = self.backbone(x)

        return self.category_head(features), self.top_15_head(features)


# CNN
# RNN @ diff # layers, different gating variants
# No video integration
# TODO: Maybe try coordinate CNNs if that seems like it'd help?
# TODO: Maybe try predicting period?
# TODO: Maybe try bidi RNNs?
# TODO: Consider normalizing by # of parameters?
def hp_model_arch():
    conditions = {
        'cnn': {'model_name': 'cnn'},
        'gru1': {'model_name': 'gru', 'num_layers': 1},
        'gru2': {'model_name': 'gru', 'num_layers': 2},
#        'gru3': {'model_name': 'gru', 'num_layers': 3}
#        'rnn1':  {'model_name': 'rnn', num_layers': 1},
#        'rnn2':  {'model_name': 'rnn', num_layers': 2},
#        'rnn3':  {'model_name': 'rnn', num_layers': 3},
#        'lstm1': {'model_name': 'lstm', num_layers': 1},
#        'lstm2': {'model_name': 'lstm', num_layers': 2},
#        'lstm3': {'model_name': 'lstm', num_layers': 3},
    }
    for expt_name, expt_args in conditions.items():
        title = f'model_arch = {expt_name}'
        path = Path('output/video_classifier/hp_model_arch')
        path.mkdir(exist_ok=True, parents=True)

        data_filename = path / f'train_log_{expt_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {expt_name}...')
            model = VideoClassifier(**expt_args)
            metrics_tracker = MetricsTracker()
            train_data, validate_data, _ = get_split_dataset()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, expt_name, title)


def hp_video_len():
    # Does the model need to see multiple cycles of a periodic pattern to pick
    # up on that information? Let's find out.
    for num_cycles in [1, 2, 3]:
        video_len = MAX_PERIOD * num_cycles
        expt_name = f'{num_cycles}x{MAX_PERIOD}'
        expt_args = {'video_len': video_len}

        title = f'video_len = {video_len}'
        path = Path('output/video_classifier/hp_video_len')
        path.mkdir(exist_ok=True, parents=True)

        data_filename = path / f'train_log_{expt_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {expt_name}...')
            model = VideoClassifier(**expt_args)
            metrics_tracker = MetricsTracker()
            train_data, validate_data, _ = get_split_dataset()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, expt_name, title)


def tune_hyperparameters():
    print('Comparing different model architectures...')
    hp_model_arch()

    # print('Comparing different video lengths...')
    # hp_video_len()


if __name__ == '__main__':
    tune_hyperparameters()

    # path = Path('output/video_classifier')
    # path.mkdir(exist_ok=True, parents=True)
    # model_filename = path / 'model.pt'

    # model = VideoClassifier()
    # metrics_tracker = MetricsTracker()
    # train_data, validate_data, test_data = get_split_dataset()

    # if model_filename.exists():
    #     print('Using pre-trained model weights')
    #     model.load_state_dict(torch.load(model_filename, weights_only=True))
    # else:
    #     print('Training model...')
    #     train_model(model, train_data, validate_data, metrics_tracker)
    #     torch.save(model.state_dict(), model_filename)
    #     metrics_tracker.get_summary().write_parquet(path / 'train_log.parquet')
    #     metrics_tracker.summarize_training(path)

    # print('Evaluating on test dataset...')
    # metrics_tracker.reset()
    # test_model(model, test_data, metrics_tracker)
    # metrics_tracker.print_summary('test')
