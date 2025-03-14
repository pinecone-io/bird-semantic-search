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




