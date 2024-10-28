from pathlib import Path
import warnings

import polars as pl
import torch
from torch import nn

from dataset import get_split_dataset
from import_data import CATEGORY_NAMES, TOP_15_NAMES
from metrics import MetricsTracker
from training import train_model, test_model

# Torch fires this warning on every call to load_state_dict()
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Borrowed from HW2.
class CNNBlock(nn.Module):
    def __init__(self, in_depth, out_depth, stride=1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            # Use circular padding, because the GOL simulations are set in a
            # toroidal space.
            nn.Conv2d(in_channels=in_depth, out_channels=out_depth,
                      kernel_size=3, padding=1, padding_mode='circular',
                      bias=False, stride=stride),
            nn.BatchNorm2d(out_depth),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # Modeled after the "StandardNet" in HW 2
        self.backbone = nn.Sequential(
            CNNBlock(1, 32, 1),       # 32x32x1
            CNNBlock(32, 64, 1),      # 32x32x32
            CNNBlock(64, 128, 2),     # 32x32x64
            CNNBlock(128, 128, 1),    # 16x16x128
            CNNBlock(128, 256, 2),    # 16x16x128
            CNNBlock(256, 256, 1),    # 8x8x256
            CNNBlock(256, 512, 2),    # 8x8x256
            CNNBlock(512, 512, 1),    # 4x4x512
            CNNBlock(512, 1024, 2),   # 4x4x512
            CNNBlock(1024, 1024, 1),  # 2x2x1024
            nn.AvgPool2d(2),          # 1x1x1024
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.category_head = nn.Sequential(
            nn.Linear(1024, len(CATEGORY_NAMES)),
            nn.Sigmoid())
        self.top_15_head = nn.Sequential(
            nn.Linear(1024, len(TOP_15_NAMES)),
            nn.Sigmoid())

    def forward(self, x):
        features = self.backbone(x)
        return self.category_head(features), self.top_15_head(features)


def hp_loss_func():
    from metrics import bce, soft_f1_score

    loss_funcs = {
        'bce': bce,
        'soft_f1_score': soft_f1_score,
    }
    for loss_name, loss_func in loss_funcs.items():
        title = f'loss = {loss_name}'
        path = Path('output/image_classifier/hp_loss_func')
        path.mkdir(exist_ok=True)

        data_filename = path / f'train_log_{loss_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {loss_name}...')
            model = ImageClassifier()
            metrics_tracker = MetricsTracker(loss_func=loss_func)
            train_data, validate_data, _ = get_split_dataset()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, loss_name, title)


def hp_task_mix():
    top_15_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for top_15_frac in top_15_fracs:
        title = f'top 15 % = {int(100 * top_15_frac)}'
        path = Path('output/image_classifier/hp_task_mix')
        path.mkdir(exist_ok=True)

        data_filename = path / f'train_log_{top_15_frac}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {int(100*top_15_frac)}% from top 15...')
            model = ImageClassifier()
            metrics_tracker = MetricsTracker()
            train_data, validate_data, _ = get_split_dataset(top_15_frac)
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, str(top_15_frac), title)


def tune_hyperparams():
    print('Comparing loss functions...')
    hp_loss_func()

    print('Comparing different task mixes...')
    hp_task_mix()


if __name__ == '__main__':
    tune_hyperparams()
    #path = Path('output/image_classifier')
    #path.mkdir(exist_ok=True)
    #model_filename = path / 'model.pt'

    #model = ImageClassifier()
    #metrics_tracker = MetricsTracker()
    #train_data, validate_data, test_data = get_split_dataset()

    #if model_filename.exists():
    #    print('Using pre-trained model weights')
    #    model.load_state_dict(torch.load(model_filename, weights_only=True))
    #else:
    #    print('Training model...')
    #    train_model(model, train_data, validate_data, metrics_tracker)
    #    torch.save(model.state_dict(), model_filename)
    #    metrics_tracker.get_summary().write_parquet(path / 'train_log.parquet')
    #    metrics_tracker.summarize_training(path)

    #print('Evaluating on test dataset...')
    #metrics_tracker.reset()
    #test_model(model, test_data, metrics_tracker)
    #metrics_tracker.print_summary('test')
