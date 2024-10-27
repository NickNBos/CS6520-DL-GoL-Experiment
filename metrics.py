import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch.nn import BCELoss
from torch.nn.functional import one_hot

from constants import NUM_EPOCHS
from import_data import CATEGORY_NAMES, TOP_15_NAMES

def soft_f1_score(true_category_oh, pred_category_oh,
                  true_pattern_id_oh, pred_pattern_id_oh):
    # To use F1 Score as a loss function, we need to make it differentiable.
    # Instead of treating category assignments as binary, instead treat a
    # predicted value between 0.0 and 1.0 as partially correct and incorrect.
    true_positives = (
        (true_category_oh * pred_category_oh).sum() +
        (true_pattern_id_oh * true_pattern_id_oh).sum())
    false_positives = (
        ((1 - true_category_oh) * pred_category_oh).sum() +
        ((1 - true_pattern_id_oh) * pred_pattern_id_oh).sum())
    false_negatives = (
        (true_category_oh * (1 - pred_category_oh)).sum() +
        (true_pattern_id_oh * (1 - pred_pattern_id_oh)).sum())
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-6)

    # We minimize loss, but want to maximize F1 Score.
    return 1.0 - f1_score


def bce(true_category_oh, pred_category_oh,
        true_pattern_id_oh, pred_pattern_id_oh):
    labels = torch.cat((true_category_oh, true_pattern_id_oh), dim=1)
    predictions = torch.cat((pred_category_oh, pred_pattern_id_oh), dim=1)
    return BCELoss()(predictions, labels)


class MetricsTracker():
    def __init__(self, summary=None, loss_func=soft_f1_score):
        self.loss_func = loss_func
        self.frames = []
        self.summary = summary

    def reset(self):
        self.frames = []
        self.summary = None

    def score_batch(self, true_category, pred_category_oh,
                    true_pattern_id, pred_pattern_id_oh, mode, epoch=None):
        true_category_oh = one_hot(
            true_category.to(torch.int64), len(CATEGORY_NAMES)
        ).squeeze().to(torch.float32)
        true_pattern_id_oh = one_hot(
            true_pattern_id.to(torch.int64), len(TOP_15_NAMES)
        ).squeeze().to(torch.float32)

        loss = self.loss_func(
            true_category_oh, pred_category_oh,
            true_pattern_id_oh, pred_pattern_id_oh)

        pred_category = torch.argmax(pred_category_oh, dim=1)
        pred_pattern_id = torch.argmax(pred_pattern_id_oh, dim=1)
        for cat_id, cat_name in enumerate(CATEGORY_NAMES):
            self.frames.append(pl.DataFrame({
                'task': 'category',
                'label': cat_name,
                'mode': mode,
                'epoch': epoch,
                'loss': loss,
                'true_positives': (
                    ((true_category == cat_id) &
                     (pred_category == cat_id))
                ).sum().item(),
                'false_positives': (
                    ((true_category != cat_id) &
                     (pred_category == cat_id))
                ).sum().item(),
                'false_negatives': (
                    ((true_category == cat_id) &
                     (pred_category != cat_id))
                ).sum().item(),
            }))
        for pattern_id, pattern_name in enumerate(TOP_15_NAMES):
            self.frames.append(pl.DataFrame({
                'task': 'top_15',
                'label': pattern_name,
                'mode': mode,
                'epoch': epoch,
                'loss': loss,
                'true_positives': (
                    ((true_pattern_id == pattern_id) &
                     (pred_pattern_id == pattern_id))
                ).sum().item(),
                'false_positives': (
                    ((true_pattern_id != pattern_id) &
                     (pred_pattern_id == pattern_id))
                ).sum().item(),
                'false_negatives': (
                    ((true_pattern_id == pattern_id) &
                     (pred_pattern_id != pattern_id))
                ).sum().item(),
            }))
        return loss

    def get_summary(self):
        if self.summary is None:
            # Gather all the data from calling ingest_batch repeatedly.
            self.summary = pl.concat(
                self.frames
            # For each class in each categorization task...
            ).group_by(
                'task', 'label', 'mode', 'epoch', maintain_order=True
            # Sum up the raw counts.
            ).agg(
                pl.col('loss').mean(),
                pl.col('true_positives').sum(),
                pl.col('false_positives').sum(),
                pl.col('false_negatives').sum(),
            # Then, compute precision, recall, and f1 score from those counts.
            ).with_columns(
                precision=(
                    pl.col('true_positives') /
                    (pl.col('true_positives') + pl.col('false_positives') + 1e-6)),
                recall=(
                    pl.col('true_positives') /
                    (pl.col('true_positives') + pl.col('false_negatives') + 1e-6)),
            ).with_columns(
                f1_score=(
                    (2 * pl.col('precision') * pl.col('recall')) /
                    (pl.col('precision') + pl.col('recall') + 1e-6))
            )
        return self.summary

    def print_summary(self, mode, file=None):
        df = self.get_summary().filter(
            # If there's more than one epoch, just look at the last one.
            (pl.col('epoch').is_null()) | (pl.col('epoch') == NUM_EPOCHS - 1) &
            # Only look at data from the this mode (train / validate / test)
            (pl.col('mode') == mode)
        )
        category_data = df.filter(
            (pl.col('task') == 'category')
        )
        print('Category macro F1 score:',
              f'{category_data["f1_score"].mean():.2f}', file=file)
        print('   Category  | Precision | Recall | F1 Score', file=file)
        print('-------------+-----------+--------+---------', file=file)
        category_data.group_by('label')
        for label in CATEGORY_NAMES:
            row_data = category_data.filter(pl.col('label') == label)
            print(f'{label:>12} | '
                  f'{row_data["precision"].item():^9.2f} | '
                  f'{row_data["recall"].item():^6.2f} | '
                  f'{row_data["f1_score"].item():^8.2f}', file=file)
        print(file=file)

        top_15_data = df.filter(pl.col('task') == 'top_15')
        print('Top-15 macro F1 score:',
              f'{top_15_data["f1_score"].mean():.2f}', file=file)
        print('Pattern Name | Precision | Recall | F1 Score', file=file)
        print('-------------+-----------+--------+---------', file=file)
        for label in TOP_15_NAMES:
            row_data = top_15_data.filter(pl.col('label') == label)
            print(f'{label:>12} | '
                  f'{row_data["precision"].item():^9.2f} | '
                  f'{row_data["recall"].item():^6.2f} | '
                  f'{row_data["f1_score"].item():^8.2f}', file=file)
        print(file=file)

    def chart_loss_curves(self, title, filename):
        df = self.get_summary().group_by(
            'mode', 'epoch', maintain_order=True
        ).agg(
            pl.col('loss').mean()
        )
        train_loss = df.filter(pl.col('mode') == 'train')['loss']
        validate_loss = df.filter(pl.col('mode') == 'validate')['loss']

        plt.figure()
        plt.suptitle(title)
        plt.gca().set_ylim([0.0, 1.0])
        plt.plot(np.arange(NUM_EPOCHS), train_loss, label='training loss')
        plt.plot(np.arange(NUM_EPOCHS), validate_loss, label='validation loss')
        plt.gca().set(xlabel='Epochs', ylabel='Loss')
        plt.legend()
        plt.savefig(filename, dpi=600)
        plt.close()

    def chart_f1_score(self, title, filename):
        df = self.get_summary().group_by(
            'task', 'epoch', maintain_order=True
        ).agg(
            pl.col('f1_score').mean()
        )
        category_f1 = df.filter(pl.col('task') == 'category')['f1_score']
        top_15_f1 = df.filter(pl.col('task') == 'top_15')['f1_score']

        plt.figure()
        plt.suptitle(title)
        plt.gca().set_ylim([0.0, 1.0])
        plt.plot(np.arange(NUM_EPOCHS), category_f1, label='category f1_score')
        plt.plot(np.arange(NUM_EPOCHS), top_15_f1, label='top_15 f1_score')
        plt.gca().set(xlabel='Epochs', ylabel='F1 Score')
        plt.legend()
        plt.savefig(filename, dpi=600)
        plt.close()

    def chart_pr_trajectory(self, title, filename):
        df = self.get_summary().group_by(
            'mode', 'epoch', maintain_order=True
        ).agg(
            pl.col('precision').mean(),
            pl.col('recall').mean()
        ).filter(
            pl.col('mode') == 'train'
        )
        precision = df['precision']
        recall = df['recall']
        plt.figure()
        plt.suptitle(title)
        plt.gca().set_xlim([0.0, 1.0])
        plt.gca().set_ylim([0.0, 1.0])
        plt.gca().set(xlabel='Precision', ylabel='Recall')
        plt.scatter(precision, recall, c=np.arange(NUM_EPOCHS))
        plt.colorbar(label='Epoch')
        plt.savefig(filename, dpi=600)
        plt.close()

    def summarize_training(self, path, suffix='', title=None):
        if len(suffix) > 0:
            suffix = f'_{suffix}'
        self.chart_loss_curves(
            title, path / f'loss{suffix}.png')
        self.chart_pr_trajectory(
            title, path / f'pr_trajectory{suffix}.png')
        self.chart_f1_score(
            title, path / f'f1_score{suffix}.png')
        with open(path / f'summary{suffix}.txt', 'w') as file:
            self.print_summary('train', file)
