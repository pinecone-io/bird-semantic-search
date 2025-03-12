# Testing the scoring metrics

import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_metrics import calculate_query_ap, calculate_mean_average_precision

# Test data setup
def get_test_dataframes():
    # should be (1 + 2/3 + 3/5) / 3 
    query_df_1 = pd.DataFrame({
        'query': ['b1', 'b1', 'b1', 'b1', 'b1'],
        'rank': [1, 2, 3, 4, 5],
        'is_relevant': [True, False, True, False, True]
    })
    query_df_ap = (1 + 2/3 + 3/5) / 3

    # should be 1 + (2/5) / 2
    query_df_2 = pd.DataFrame(
        {
            'query': ['b2', 'b2', 'b2', 'b2', 'b2'],
            'rank': [1, 2, 3, 4, 5],
            'is_relevant': [True, False, False, False, True]
        }
    )
    query_df_ap_2 = (1 + (2/5)) / 2

    all_df = pd.concat([query_df_1, query_df_2])
    mean_average_precision = (query_df_ap + query_df_ap_2) / 2
    
    return query_df_1, query_df_ap, query_df_2, query_df_ap_2, all_df, mean_average_precision

def test_calculate_query_ap():
    query_df_1, query_df_ap, query_df_2, query_df_ap_2, _, _ = get_test_dataframes()
    
    calc_ap = calculate_query_ap(query_df_1)
    assert calc_ap == query_df_ap, f"Expected AP: {query_df_ap}, Got: {calc_ap}"

    calc_ap_2 = calculate_query_ap(query_df_2)
    assert calc_ap_2 == query_df_ap_2, f"Expected AP 2: {query_df_ap_2}, Got: {calc_ap_2}"

def test_calculate_mean_average_precision():
    _, _, _, _, all_df, mean_average_precision = get_test_dataframes()
    
    calc_map = calculate_mean_average_precision(all_df)
    assert calc_map == mean_average_precision, f"Expected MAP: {mean_average_precision}, Got: {calc_map}"
