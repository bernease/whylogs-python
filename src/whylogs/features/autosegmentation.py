from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


def _entropy(df: Union[pd.DataFrame],
             current_split_columns: List[Optional[str]],
             column_name: str,
             target_column_name : str = None,
             normalized: bool = False):
    """Entropy calculation"""
    total_entropy = 0

    col_groups = df.groupby(current_split_columns)
    for col_value, col_group in col_groups:
        probs = col_group[column_name].value_counts(normalize=True,
                                                    dropna=False)
        entropy = -np.sum([p_i * np.log(p_i) for p_i in probs])
        if normalized:
            entropy /= np.log(len(col_group[column_name].unique()))

        total_entropy += entropy

    return total_entropy


def _information_gain_ratio(df: pd.DataFrame,
                            current_split_columns: List[Optional[str]],
                            column_name: str,
                            target_column_name: str,
                            normalized: bool = False):
    """Information gain ratio"""
    ratios = []

    n_rows = df.shape[0]

    col_entropy = 0
    split_entropy = 0
    col_groups = df.groupby(current_split_columns)

    for col_value, col_group in col_groups:
        col_probs = col_group.value_counts() / n_rows
        col_entropy += col_probs.sum() * -np.sum(
                [p_i * np.log(p_i) for p_i in col_probs])

        # For gain ratio, we must divide by entropy of ratio of |S_v| / |S|
        split_entropy += len(col_group) / n_rows * \
                np.log(len(col_group) / n_rows)

    ratios.append(-(_entropy(df, target_column_name) -
                    col_entropy) / split_entropy)

    assert len(ratios) == len
    return ratios


def _find_best_split(df: pd.DataFrame,
                     current_split_columns: List[Optional[str]],
                     valid_column_names: Union[str],
                     target_column_name: Optional[str] = None):
    if target_column_name is None:
        score_fn = _entropy
    else:
        score_fn = _information_gain_ratio

    max_score_tuple = 0, None
    for column_name in valid_column_names:
        value = score_fn(df, column_name, target_column_name)
        if value > max_score_tuple[0]:
            max_score_tuple = value, column_name

    print(f"result from _find_best_split: {max_score_tuple}")
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

        if target_field in valid_column_names:
            valid_column_names.remove(target_field)

        if not valid_column_names and not segments:
            print("\nFound zero valid columns.")

        _, segment_column_name = _find_best_split(
                df,
                [target_field],
                valid_column_names,
                target_column_name=target_field
        )
        segments.append(segment_column_name)
        segments_used *= len(df[segment_column_name].unique())

    return segments
