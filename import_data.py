from urllib import request
from pathlib import Path

import polars as pl

# The Catagolue has many different object collections produced by different
# random searches. For now, we just pull from the biggest collection, with the
# object sub-categories hard coded. To get even more objects, we could read the
# index pages from many searches and scrape their size and period lists to
# download everything. Not doing that for now, since there's a high degree of
# redundancy (most searches find the same objects, again and again)
URL_PREFIX ='https://catagolue.hatsya.com/textcensus/b3s23/C1'

OUTPUT_PATH = Path('catagolue.parquet')


def fetch_csv(url):
    with request.urlopen(url) as csv_data:
        return pl.read_csv(csv_data)


def import_still_lifes():
    sizes = [
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 50, 56
    ]
    return pl.concat([
        fetch_csv(
            f'{URL_PREFIX}/xs{size}'
        ).with_columns(still_life=True, size=size)
        for size in sizes
    ])


def import_oscillators():
    periods = [ 2, 3, 4, 5, 6, 8, 14, 15, 16, 24, 30, 46, 120 ]
    return pl.concat([
        fetch_csv(
            f'{URL_PREFIX}/xp{period}'
        ).with_columns(oscillator=True, period=period)
        for period in periods
    ])


def import_spaceships():
    periods = [ 4, 7, 12, 16 ]
    return pl.concat([
        fetch_csv(
            f'{URL_PREFIX}/xq{period}'
        ).with_columns(spaceship=True, period=period)
        for period in periods
    ])


def import_all():
    return pl.concat([
        import_still_lifes(),
        import_oscillators(),
        import_spaceships()
    ], how='diagonal')


def process(df):
    return df.with_columns(
        # For all category columns, make sure they're boolean typed and replace
        # empty values with False.
        pl.col('still_life').cast(bool).fill_null(False),
        pl.col('oscillator').cast(bool).fill_null(False),
        pl.col('spaceship').cast(bool).fill_null(False),

        # For all feature columns, make sure they're integer typed. For size,
        # null values make sense, since we only have data for the still lifes.
        # For period, the only null values should come from the still lifes,
        # which logically have a period of 0.
        pl.col('size').cast(pl.Int64),
        pl.col('period').cast(pl.Int64).fill_null(0),

        # TODO: Maybe store rendered patterns here and compute basic metrics
        # from them, like the shape of the rendered pattern?
    )


if __name__ == '__main__':
    if not OUTPUT_PATH.exists():
        print('Downloading pattern data from Catagolue...')
        df = import_all()
        print('Done.')
    else:
        print('Using cached Catagolue data.')
        df = pl.read_parquet(OUTPUT_PATH)
    print('Processing data...')
    df = process(df)
    df.write_parquet(OUTPUT_PATH)
    print('Done')
