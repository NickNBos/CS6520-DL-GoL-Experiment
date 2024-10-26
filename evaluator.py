import polars as pl

from import_data import CATEGORY_NAMES, TOP_15_NAMES


class Evaluator():
    def __init__(self):
        self.frames = []

    def ingest_batch(self, true_category, pred_category,
                     true_pattern_id, pred_pattern_id):
        for cat_id, cat_name in enumerate(CATEGORY_NAMES):
            self.frames.append(pl.DataFrame({
                'task': 'category',
                'label': cat_name,
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

    def get_summary(self):
        # Gather all the data from calling ingest_batch repeatedly.
        return pl.concat(
            self.frames
        # For each class in each categorization task...
        ).group_by(
            'task', 'label'
        # Sum up the raw counts.
        ).agg(
            pl.col('true_positives').sum(),
            pl.col('false_positives').sum(),
            pl.col('false_negatives').sum()
        # Then, compute precision, recall, and f1 score from those counts.
        ).with_columns(
            precision=(
                pl.col('true_positives') /
                (pl.col('true_positives') + pl.col('false_positives'))),
            recall=(
                pl.col('true_positives') /
                (pl.col('true_positives') + pl.col('false_negatives'))),
        ).with_columns(
            f1_score=(
                (2 * pl.col('precision') * pl.col('recall')) /
                (pl.col('precision') + pl.col('recall')))
        ).fill_nan(0.0)

    def print_summary(self):
        df = self.get_summary()
        category_data = df.filter(
            pl.col('task') == 'category'
        )
        print('Category macro F1 score:',
              f'{category_data["f1_score"].mean():.2f}')
        print('Category   | Precision | Recall | F1 Score')
        print('-----------+-----------+--------+---------')
        category_data.group_by('label')
        for label in CATEGORY_NAMES:
            row_data = category_data.filter(pl.col('label') == label)
            print(f'{label:>10} | '
                  f'{row_data["precision"].item():^9.2f} | '
                  f'{row_data["recall"].item():^6.2f} | '
                  f'{row_data["f1_score"].item():^8.2f}')
        print()

        top_15_data = df.filter(pl.col('task') == 'top_15')
        print('Top-15 macro F1 score:',
              f'{top_15_data["f1_score"].mean():.2f}')
        print('Pattern Name | Precision | Recall | F1 Score')
        print('-------------+-----------+--------+---------')
        for label in TOP_15_NAMES:
            row_data = top_15_data.filter(pl.col('label') == label)
            print(f'{label:>12} | '
                  f'{row_data["precision"].item():^9.2f} | '
                  f'{row_data["recall"].item():^6.2f} | '
                  f'{row_data["f1_score"].item():^8.2f}')
        print()
