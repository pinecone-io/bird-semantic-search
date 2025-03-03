import pandas as pd

def get_unique_relevant_birds(df):
    """Get count of unique relevant birds"""
    if df.empty:
        return 0
    relevant_birds = df[df['is_relevant']]['bird'].unique()
    return int(len(relevant_birds))

def calculate_mean_average_precision(df):
    """Calculate mean average precision across queries. This measures how relevant the results are, relative to the rank of the result."""
    if df.empty:
        return 0.0
        
    ap_values = []
    query_groups = df.groupby('query')
    
    for _, group in query_groups:
        query_ap = calculate_query_ap(group)
        if query_ap > 0:
            ap_values.append(query_ap)
    print(ap_values)
            
    return float(sum(ap_values) / len(ap_values)) if ap_values else 0.0

def calculate_query_ap(query_df):
    """Calculate average precision for a single query"""
    query_relevant_df = query_df[query_df['is_relevant']].sort_values('rank')
    if query_relevant_df.empty:
        return 0.0
        
    precisions = []
    for _, row in query_relevant_df.iterrows():
        rank = row['rank']
        hits_to_rank = query_df[query_df['rank'] <= rank]
        precision_at_k = hits_to_rank['is_relevant'].sum() / len(hits_to_rank)
        precisions.append(float(precision_at_k))
        
    return sum(precisions) / len(precisions) if precisions else 0.0



# Sanity checks on metrics

# Average precision

# should be (1 + 2/3 + 3/5) / 3 
query_df_1 = pd.DataFrame({
    'query': ['b1', 'b1', 'b1', 'b1', 'b1'],
    'rank': [1, 2, 3, 4, 5],
    'is_relevant': [True, False, True, False, True]
})
query_df_ap = (1 + 2/3 + 3/5) / 3

# should be 1 + (2/5) / 5
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

if __name__ == "__main__":
    calc_ap = calculate_query_ap(query_df_1)
    if calc_ap != query_df_ap:
        print(f"Expected AP: {query_df_ap}, Got: {calc_ap}")
    assert calc_ap == query_df_ap

    calc_ap_2 = calculate_query_ap(query_df_2) 
    if calc_ap_2 != query_df_ap_2:
        print(f"Expected AP 2: {query_df_ap_2}, Got: {calc_ap_2}")
    assert calc_ap_2 == query_df_ap_2

    calc_map = calculate_mean_average_precision(all_df)
    if calc_map != mean_average_precision:
        print(f"Expected MAP: {mean_average_precision}, Got: {calc_map}")
    assert calc_map == mean_average_precision

