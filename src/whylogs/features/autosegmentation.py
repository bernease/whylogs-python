from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


def _entropy(df, column_name, target_column_name=None):
    """Entropy calculation"""
    probs = df[column_name].value_counts(normalize=True, dropna=False)
    return -np.sum([p_i * np.log(p_i) for p_i in probs])


def _normalized_entropy(df, column_name, target_column_name=None):
    """Normalized entropy calculation that mitigates bias toward high cardinality variables"""
    return _entropy(df, column_name, target_column_name=None) / np.log(
            len(df[column_name].unique()))


def _information_gain_ratio(df, column_name, target_column_name):
    """Information gain ratio """
    n_rows = df.shape[0]

    col_entropy = 0
    split_entropy = 0
    col_groups = df[[target_column_name, column_name]].groupby(
            target_column_name, dropna=False)
    for col_value, col_group in col_groups:
        col_probs = col_group.value_counts() / n_rows
        col_entropy += col_probs.sum() * -np.sum(
                [p_i * np.log(p_i) for p_i in col_probs])

        # For gain ratio, we must divide by entropy of ratio of |S_v| / |S|
        split_entropy += len(col_group) / n_rows * np.log(
                len(col_group) / n_rows)

    return -(_entropy(df, target_column_name) - col_entropy) / split_entropy


def _find_best_split(df, column_names, target_column_name=None):
    if target_column_name == None:
        score_fn = _normalized_entropy
    elif target_column_name in df.columns:
        if target_column_name in column_names:
            column_names.remove(target_column_name)
        score_fn = _information_gain_ratio
    else:
        print(f"\nIncorrect target_field parameter: {target_column_name}")

    max_score_tuple = 0, None
    for column_name in column_names:
        value = score_fn(df, column_name, target_column_name)
        if value > max_score_tuple[0]:
            max_score_tuple = value, column_name

    return max_score_tuple


def estimate_segments(
    df: pd.DataFrame,
    target_field: str = None,
    max_segments: int = 30
) -> Optional[Union[List[Dict], List[str]]]:
    """
    Estimates the most important features and values on which to segment
    data profiling using entropy-based methods.

    :param df: the dataframe of data to profile
    :param target_field: target field (optional)
    :param max_segments: upper threshold for total combinations of segments,
    default 30
    :return: a list of segmentation feature names
    """
    segments = []
    segments_used = 1

    while segments_used < max_segments:

        valid_column_names = []

        for col in df.columns:
            n_unique = len(df[col].unique())
            nulls = df[col].isnull().value_counts(normalize=True)
            null_perc = 0.0 if True not in nulls.index else nulls[True]
            if (n_unique > 1 and
                    n_unique * segments_used <= max_segments - segments_used and
                    col not in segments and
                    null_perc <= 0.2):
                valid_column_names.append(col)

        if not valid_column_names:
            if not segments:
                print("\nFound zero valid columns.")

        _, segment_column_name = _find_best_split(df, valid_column_names,
                                                  target_column_name=target_field)
        segments.append(segment_column_name)
        segments_used *= len(df[segment_column_name].unique())

    return segments
