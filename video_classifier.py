from pathlib import Path
import warnings

import polars as pl
import torch
from torch import nn
from constants import VIDEO_LEN

from dataset import get_split_dataset
from image_classifier import ImageClassifier
from metrics import MetricsTracker
from simulate import simulate_batch
from training import train_model, test_model

# Torch fires this warning on every call to load_state_dict()
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class VideoClassifier(nn.Module):
    def __init__(self, image_classifier, freeze_features=False, model_name=''):
        super(VideoClassifier, self).__init__()
        self.image_classifier = image_classifier
        if freeze_features:
            # Load pre-trained weights into our image classifier
            image_classifier.load_state_dict(
                torch.load('output/image_classifier/model.pt',
                           weights_only=True))
            # Freeze the feature weights so we only train the classification
            # parts.
            for param in self.image_classifier.backbone.parameters():
                param.requires_grad = False

        # TODO: Implement different temporal models...
        if model_name == '2+1D CNN':
            ...
        elif model_name == 'CNN+GRU':
            ...

        # Reuse the classification heads from the image model (note, they will
        # get retrained from scratch). Note, we could also implement
        # video-specific classification heads, if we need to!
        self.category_head = image_classifier.category_head
        self.top_15_head = image_classifier.top_15_head

    def forward(self, x):
        # Take a batch of initial states and simulate them all in parallel.
        x = simulate_batch(x, VIDEO_LEN)
        batch_size, steps, num_channels, height, width = x.shape

        # Pretend like each frame is just another sample in a larger batch, and
        # run the image classifier on each frame that way. Then, split it back
        # into a one set of 1024 features per frame per batch item.
        x = x.reshape(batch_size * steps, num_channels, height, width)
        features = self.image_classifier.backbone(x)
        assert features.shape == (batch_size * steps, 1024)
        features = features.reshape(batch_size, steps, 1024)

        # TODO: Some sort of temporal processing goes here. For now, we just
        # take the average prediction from each frame as a placeholder.
        features = features.mean(dim=1)
        return self.category_head(features), self.top_15_head(features)


def hp_model_arch():
    # TODO: What variations do we want to try?
    conditions = {
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
            # TODO: Pass in different configurations to the image and video
            # classifiers.
            image_classifier = ImageClassifier()
            model = VideoClassifier(image_classifier)
            metrics_tracker = MetricsTracker()
            train_data, validate_data, _ = get_split_dataset()
            train_model(model, train_data, validate_data, metrics_tracker)
            metrics_tracker.get_summary().write_parquet(data_filename)

        metrics_tracker.summarize_training(path, expt_name, title)


if __name__ == '__main__':
    path = Path('output/video_classifier')
    path.mkdir(exist_ok=True, parents=True)
    model_filename = path / 'model.pt'

    image_classifier = ImageClassifier('minimal')
    model = VideoClassifier(image_classifier)
    metrics_tracker = MetricsTracker()
    train_data, validate_data, test_data = get_split_dataset()

    if model_filename.exists():
        print('Using pre-trained model weights')
        model.load_state_dict(torch.load(model_filename, weights_only=True))
    else:
        print('Training model...')
        train_model(model, train_data, validate_data, metrics_tracker)
        torch.save(model.state_dict(), model_filename)
        metrics_tracker.get_summary().write_parquet(path / 'train_log.parquet')
        metrics_tracker.summarize_training(path)

    print('Evaluating on test dataset...')
    metrics_tracker.reset()
    test_model(model, test_data, metrics_tracker)
    metrics_tracker.print_summary('test')
