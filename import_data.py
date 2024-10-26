from urllib import request
from pathlib import Path

import numpy as np
import polars as pl

from decode import Decoder
import generate_fizzlers as gfz

# The Catagolue has many different object collections produced by different
# random searches. For now, we just pull from the biggest collection, with the
# object sub-categories hard coded. To get even more objects, we could read the
# index pages from many searches and scrape their size and period lists to
# download everything. Not doing that for now, since there's a high degree of
# redundancy (most searches find the same objects, again and again)
CENSUS_PREFIX ='https://catagolue.hatsya.com/textcensus/b3s23/C1'

OUTPUT_PATH = Path('catagolue.parquet')

# Names for the 15 most common patterns, which account for 99.9% of objects
# found in random soups.
TOP_15 = {
    'xs4_33': 'block',
    'xs5_253': 'boat',
    'xs7_2596': 'loaf',
    'xq4_153': 'glider',
    'xs6_696': 'beehive',
    'xp2_7': 'blinker',
    'xs6_356': 'ship',
    'xs4_252': 'tub',
    'xs8_6996': 'pond',
    'xs7_25ac': 'long boat',
    'xp2_7e': 'toad',
    'xs12_g8o653z11': 'ship-tie',
    'xp2_318c': 'beacon',
    'xs6_25a4': 'barge',
    'xs14_g88m952z121': 'half-bakery',
}
TOP_15_NAMES = list(TOP_15.values()) + ['other']

CATEGORY_NAMES = ['still_life', 'oscillator', 'spaceship', 'other']


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
            f'{CENSUS_PREFIX}/xs{size}'
        ).with_columns(
            category=CATEGORY_NAMES.index('still_life'),
            size=size,
            period=1
        )
        for size in sizes
    ])


def import_oscillators():
    periods = [ 2, 3, 4, 5, 6, 8, 14, 15, 16, 24, 30, 46, 120 ]
    return pl.concat([
        fetch_csv(
            f'{CENSUS_PREFIX}/xp{period}'
        ).with_columns(
            category=CATEGORY_NAMES.index('oscillator'),
            period=period
        )
        for period in periods
    ])


def import_spaceships():
    periods = [ 4, 7, 12, 16 ]
    census_spaceships = pl.concat([
        fetch_csv(
            f'{CENSUS_PREFIX}/xq{period}'
        ).with_columns(
            category=CATEGORY_NAMES.index('spaceship'),
            period=period
        )
        for period in periods
    ])

    # The main census we use has very few spaceships, and they're almost all
    # period 4. To get a more diverse set of samples, we also use a supplental
    # dataset with wider coverage.
    periods = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 30, 96]
    extra_spaceships = pl.concat([
        fetch_csv(
            f'https://catagolue.hatsya.com/textcensus/b3s23/shipthread_stdin/xq{period}'
        ).with_columns(
            category=CATEGORY_NAMES.index('spaceship'),
            period=period
        )
        for period in periods
    ])

    # Merge the data from the two sources.
    return census_spaceships.join(
        extra_spaceships,
        on=['apgcode', 'category', 'period'],
        how='full',
        coalesce=True
    ).with_columns(
        # The occurrence columns don't match between the two data sources, so
        # make a new column by summing them.
        occurrences=(
            pl.col('occurrences').fill_null(0) +
            pl.col('occurrences_right').fill_null(0)
        )
    ).drop(
        # Drop the occurrence data from the second data source.
        'occurrences_right'
    )


def create_fizzlers(sizes = [6], goal = 500, tries = 5000):
    
    fizzler_list = []
    for size in sizes:
        fizzler_list.extend(gfz.generate(size, goal, tries))
    
    patterns = []
    lifespans = []
    for fizzler in fizzler_list:
        # Another redundancy to prevent duplicates
        # as the padding can throw off the size enforcement
        if fizzler[0] not in patterns:
            patterns.append(fizzler[0])
            lifespans.append(fizzler[1])
    
    # Cop some extra info
    sizes = [ np.count_nonzero(pattern) for pattern in patterns ]
    heights = [ len(pattern) for pattern in patterns ]
    widths = [ len(pattern[0]) for pattern in patterns ]
    
    categories = [CATEGORY_NAMES.index('other')] * len(patterns)
    
    data = {
        "category":categories,
        "pattern":patterns,
        "lifespan":lifespans,
        "size":sizes,
        "width":widths,
        "height":heights,
        "top_15":TOP_15_NAMES.index('other')
    }
    
    # Janky workaround for ensuring catagory is the same dtype (i32)
    df = pl.DataFrame(data).with_columns(pl.col('category').cast(pl.Int32))
    
    return df
    
def import_all():
    return pl.concat([
        import_still_lifes(),
        import_oscillators(),
        import_spaceships()
    ], how='diagonal')



def process(df):
    # First, remove all old fizzlers
    df = df.filter(pl.col("category") != CATEGORY_NAMES.index('other'))
    
    # Pre-render all the patterns from their apgcodes.
    decoder = Decoder()
    # The problem here is that polars doesn't play nice with numpy arrays,
    # so the arrays must first be translated into lists
    # Luckily, it seems that the sizes can be incongruent at least
    patterns = [
        decoder.standard_one_pad(decoder.decode(apgcode)).tolist()
        for apgcode in df['apgcode']
    ]

    # Compute metadata about each pattern.
    # TODO: It might be better to find the maximum size, width, and height of a
    # pattern over its full period.
    sizes = [ np.count_nonzero(pattern) for pattern in patterns ]
    heights = [ len(pattern) for pattern in patterns ]
    widths = [ len(pattern[0]) for pattern in patterns ]

    # Regenerate all computed columns
    return df.drop(
        'top_15', 'pattern', 'size', 'width', 'height'
    ).join(
        # Add names for the top-15 most recognized patterns. Rare patterns are
        # marked by a null value.
        pl.DataFrame({
            'apgcode': TOP_15.keys(),
            'top_15': np.arange(15)
        }), on='apgcode', how='left'
    ).with_columns(
        # Make sure any object that isn't in the top 15 is categorized as
        # 'other' rather than null, for consistency.
        pl.col('top_15').fill_null(TOP_15_NAMES.index('other')),
        pl.Series(name='pattern', values=patterns),
        pl.Series(name='size', values=sizes),
        pl.Series(name='width', values=widths),
        pl.Series(name='height', values=heights),
    )
        
def regenerate_catagolue_data():
    if not OUTPUT_PATH.exists():
        print('Downloading pattern data from Catagolue...')
        df = import_all()
        print('Done.')
    else:
        print('Using cached Catagolue data.')
        df = pl.read_parquet(OUTPUT_PATH)
        print(df.columns)
    print('Processing data...')
    df = process(df)
    
    print('Adding fizzlers...')
    df = pl.concat([
        create_fizzlers(),
        df
    ], how='diagonal')

    df.write_parquet(OUTPUT_PATH)
    print('Done')
    
    # Debug print to determine how many of each type exist
    # I believe it is as follows:
        # 0: Still-life
        # 1: Oscillator
        # 2: Spaceship
        # 3: Fizzler
    # print(df['category'].value_counts())
    
    return df


def load_catagolue_data():
    if OUTPUT_PATH.exists():
        return pl.read_parquet(OUTPUT_PATH)
    else:
        return regenerate_catagolue_data()


if __name__ == '__main__':
    df = regenerate_catagolue_data()
