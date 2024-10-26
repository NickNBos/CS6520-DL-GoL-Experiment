from pathlib import Path
import warnings

import torch
from torch import nn

from dataset import get_split_dataset
from import_data import CATEGORY_NAMES, TOP_15_NAMES
from training import train_model, test_model, chart_loss_curves

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
        output_size = len(CATEGORY_NAMES) + len(TOP_15_NAMES)
        # Modeled after the "StandardNet" in HW 2
        self.net = nn.Sequential(
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
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    path = Path('output/image_classifier')
    path.mkdir(exist_ok=True)
    model_filename = path / 'model.pt'

    model = ImageClassifier()
    train_data, validate_data, test_data = get_split_dataset()

    if model_filename.exists():
        print('Using pre-trained model weights')
        model.load_state_dict(torch.load(model_filename, weights_only=True))
    else:
        print('Training model...')
        chart_loss_curves(
            *train_model(model, train_data, validate_data), path / 'loss.png')
        torch.save(model.state_dict(), model_filename)

    print('Evaluating on test dataset...')
    test_model(model, test_data)
