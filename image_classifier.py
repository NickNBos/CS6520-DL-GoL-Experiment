from pathlib import Path
import warnings

import polars as pl
import torch
from torch import nn, optim
from torchinfo import summary
from torchvision.models import vgg13, VGG13_Weights
from constants import BATCH_SIZE, WORLD_SIZE

from dataset import get_split_dataset
from import_data import CATEGORY_NAMES, TOP_15_NAMES
from metrics import MetricsTracker, bce, focal, soft_f1_score, compare_runs
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


# Model variants to try:
#  - "StandardNet" from HW2
#  - Off-the-shelf VGG (pre-trained and from scratch)
#  - Minimal CNN
#  - With and without coordinates embedded
class ImageClassifier(nn.Module):
    def __init__(self, model_name='standard'):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name

        if model_name == 'standard':
            # Modeled after the "StandardNet" in HW 2
            self.backbone = nn.Sequential(
                CNNBlock(1, 32, 1),      # 32x32x1
                CNNBlock(32, 64, 1),     # 32x32x32
                CNNBlock(64, 128, 2),    # 32x32x64
                CNNBlock(128, 128, 1),   # 16x16x128
                CNNBlock(128, 256, 2),   # 16x16x128
                CNNBlock(256, 256, 1),   # 8x8x256
                CNNBlock(256, 512, 2),   # 8x8x256
                CNNBlock(512, 512, 1),   # 4x4x512
                CNNBlock(512, 1024, 2),  # 4x4x512
                CNNBlock(1024, 1024, 1), # 2x2x1024
                nn.AvgPool2d(2),         # 1x1x1024
                nn.Flatten(),
                nn.Linear(1024, 1024),
                nn.ReLU()
            )
        if model_name == 'minimal':
            # A tiny CNN.
            self.backbone = nn.Sequential(
                CNNBlock(1, 32, 1),  # 32x32x1
                nn.AvgPool2d(2),     # 16x16x32
                CNNBlock(32, 64, 1), # 16x16x32
                nn.AvgPool2d(2),     # 8x8x64
                nn.Flatten(),
                nn.Linear(8 * 8 * 64, 1024),
                nn.ReLU()
            )

        elif model_name == 'vgg' or model_name == 'vgg_pretrained':
            if model_name == 'vgg_pretrained':
                weights = VGG13_Weights.DEFAULT
            else:
                weights = None
            self.vgg = vgg13(weights)
            if weights:
                # If we're using pre-trained weights, then freeze the features
                # sub-network so that we're only fine-tuning the weights
                # relevant to classification.
                for param in self.vgg.features.parameters():
                    param.requires_grad = False
            self.backbone = nn.Sequential(
                self.vgg.features,
                self.vgg.avgpool,
                # An adapter from the VGG intermediate modules to our custom
                # classifier heads. This is NOT frozen, so it allows us to
                # transform the default features before feeding them into the
                # classifier heads.
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(),
            )

        # All backbone models produce a feature vector of size 1024. Here we
        # define classifier heads for both tasks.
        self.category_head = nn.Sequential(
            nn.Linear(1024, len(CATEGORY_NAMES)))
        self.top_15_head = nn.Sequential(
            nn.Linear(1024, len(TOP_15_NAMES)))

    def forward(self, x):
        # VGG expects three channel inputs, but our images are single channel.
        # So, just repeat that one channel three times.
        if self.model_name.startswith('vgg'):
            x = x.expand(-1, 3, WORLD_SIZE, WORLD_SIZE)

        features = self.backbone(x)
        return self.category_head(features), self.top_15_head(features)


def hp_loss_func():
    path = Path('output/image_classifier/hp_loss_func')
    path.mkdir(exist_ok=True, parents=True)

    loss_funcs = {
        'focal': focal,
        'bce': bce,
        'soft_f1_score': soft_f1_score,
    }
    for loss_name, loss_func in loss_funcs.items():
        title = f'loss = {loss_name}'
        data_filename = path / f'train_log_{loss_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {loss_name}...')
            model = ImageClassifier()
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker(loss_func=loss_func)
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, loss_name, title)
    compare_runs(path, loss_funcs.keys())


def hp_optimizer():
    path = Path('output/image_classifier/hp_optimizer')
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
            model = ImageClassifier()
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker, optimizer)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, optim_name, title)
    compare_runs(path, optimizers.keys())


def hp_task_mix():
    path = Path('output/image_classifier/hp_task_mix')
    path.mkdir(exist_ok=True, parents=True)

    top_15_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    for top_15_frac in top_15_fracs:
        title = f'top 15 % = {int(100 * top_15_frac)}'
        data_filename = path / f'train_log_{top_15_frac}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {int(100*top_15_frac)}% from top 15...')
            model = ImageClassifier()
            train_data, validate_data, _ = get_split_dataset(top_15_frac)
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, str(top_15_frac), title)
    compare_runs(path, map(str, top_15_fracs))


def hp_model_arch():
    path = Path('output/image_classifier/hp_model_arch')
    path.mkdir(exist_ok=True, parents=True)

    # TODO: These "_with_coords" variants are inspired by CoordinateCNNs. They
    # make it possible to model spatial details even though CNNs are normally
    # translation invariant. I suspect this won't help for the image
    # classifier, but may help with the video classifier, if it lets the model
    # reason about the spatial offsets between features in different frames.
    model_names = ['minimal', 'vgg_pretrained', 'vgg', 'standard']
    for model_name in model_names:
        title = f'model_arch = {model_name}'
        data_filename = path / f'train_log_{model_name}.parquet'
        if data_filename.exists():
            print('Regenerating outputs from cached data...')
            metrics_tracker = MetricsTracker(pl.read_parquet(data_filename))
        else:
            print(f'Training with {model_name}...')
            model = ImageClassifier(model_name)
            train_data, validate_data, _ = get_split_dataset()
            metrics_tracker = MetricsTracker()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)
            with open(path / f'model_summary_{model_name}.txt', 'w') as file:
                print(summary(model, verbose=0), file=file)

        metrics_tracker.summarize_training(path, model_name, title)
    compare_runs(path, model_names)


def tune_hyperparams():
    # print('Comparing loss functions...')
    # hp_loss_func()

    # print('Comparing different optimizers...')
    # hp_optimizer()

    # print('Comparing different task mixes...')
    # hp_task_mix()

    print('Comparing different model architectures...')
    hp_model_arch()


def get_model():
    model = ImageClassifier()
    model.load_state_dict(
        torch.load('output/image_classifier/model.pt', weights_only=True))
    return model


if __name__ == '__main__':
    # tune_hyperparams()

    path = Path('output/image_classifier')
    path.mkdir(exist_ok=True, parents=True)
    model_filename = path / 'model.pt'

    model = ImageClassifier()
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
