import re
from collections import Counter
from functools import cache
from tempfile import mkdtemp
from typing import Callable, cast

import pandas as pd
from nltk.corpus import stopwords
from nltk.downloader import download as _nltk_download
from nltk.tokenize import word_tokenize
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, lit

IS_DELIMITED_RE = re.compile(r'[^,;\|]+([,;\|][^,;\|]+)+')
DELIMITER_RE = re.compile(r'(,|;|\|)')

URL_RE = re.compile(
    r'(http|ftp|https):\/\/'
    r'([\w_-]+(?:(?:\.[\w_-]+)+))'
    r'([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
)

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b')

_LEGACY_NAME_MAP = {
    'sample_0': 'sample_1',
    'sample_1': 'sample_2',
    'sample_2': 'sample_3',
    'sample_3': 'sample_4',
    'sample_4': 'sample_5',
    'is_delimited': 'has_delimiters',
    'is_url': 'has_url',
    'is_email': 'has_email',
    'is_datetime': 'has_date',
    'word_count': 'word_count',
    'stopword_count': 'stopword_total',
    'char_count': 'char_count',
    'whitespace_count': 'whitespace_count',
    'delimiter_count': 'delim_count',
    'is_list': 'is_list',
    'is_long_sentence': 'is_long_sentence',
}

LEGACY_NAME_MAP = {}

for k, v in _LEGACY_NAME_MAP.items():
    if k.endswith('count'):
        LEGACY_NAME_MAP[f'mean_{k}'] = f'mean_{v}'
        LEGACY_NAME_MAP[f'std_{k}'] = f'stdev_{v}'
    else:
        LEGACY_NAME_MAP[k] = v

LEGACY_NAME_MAP['std_word_count'] = 'std_dev_word_count'


@cache
def _stopwords():
    tmpdir = mkdtemp()
    _nltk_download('stopwords', download_dir=tmpdir)
    _nltk_download('punkt', download_dir=tmpdir)

    return frozenset(stopwords.words('english'))


def _stopword_count(s: str) -> int:
    counts = Counter(word_tokenize(s))
    return sum(v for k, v in counts.items() if k in _stopwords())


def sample_features(
    df: pd.DataFrame,
    /,
    *,
    use_legacy_names=False,
) -> pd.DataFrame:
    """
    Extracts features from tidy Pandas DataFrame.

    Expects columns `column`, `value`.
    """
    # COLUMN
    values: pd.Series = df['value']
    # NOTE: Bug in original means that delimiter_count will always equal whitespace_count
    df['is_delimited'] = values.str.match(IS_DELIMITED_RE)
    df['delimiter_count'] = values.str.count(DELIMITER_RE)
    df['word_count'] = values.str.split(' ', regex=False).map(len)
    df['char_count'] = values.str.len()
    df['whitespace_count'] = values.str.count('')
    df['is_url'] = values.str.match(URL_RE)
    df['is_email'] = values.str.match(EMAIL_RE)
    df['is_datetime'] = pd.to_datetime(values, errors='coerce').notnull()

    df['stopword_count'] = values.map(_stopword_count)

    aggs: dict[str, str | Callable] = {'value': list}
    is_cols = [
        'is_delimited',
        'is_url',
        'is_email',
        'is_datetime',
    ]
    count_cols = [
        'delimiter_count',
        'word_count',
        'char_count',
        'whitespace_count',
        'stopword_count',
    ]
    cols = is_cols + count_cols
    aggs.update({col: 'sum' for col in cols})

    result = df.groupby('column').agg(aggs)

    for col in is_cols:
        result[col] = result[col] >= 3  # type: ignore

    for col in count_cols:
        s = result[col]
        result[f'mean_{col}'] = s.mean()
        result[f'std_{col}'] = s.std()

    result = result.drop(columns=count_cols)

    result['is_list'] = result.is_delimited & (result.mean_char_count < 100)
    result['is_long_sentence'] = result.mean_word_count > 10

    for i in range(5):
        result[f'sample_{i}'] = values.str.get(i)

    result: pd.DataFrame = cast(
        pd.DataFrame,
        result[list(LEGACY_NAME_MAP.keys())],
    )

    if use_legacy_names:
        result = result.rename(columns=LEGACY_NAME_MAP)

    return result


def _sample_with_select_distinct(
    df: SparkDataFrame,
    n: int = 5,
) -> SparkDataFrame:
    cols = df.columns

    it = iter(
        df.select(
            lit(name).alias('column'), col(name).cast('string').alias('value')
        )
        .distinct()
        .limit(n)
        for name in cols
    )

    result = next(it)
    for expr in it:
        result = result.union(expr)

    return result
