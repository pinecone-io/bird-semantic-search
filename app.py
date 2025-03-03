import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
import altair as chart
from datetime import datetime
import nltk

#loading to enable bm25
nltk.download('punkt_tab')


from query_db import *
from search_metrics import *

# Load metadata for images
with open("parsed_birds/parsing_metadata.json", 'r') as f:
    parsing_metadata = json.load(f)

st.title("Bird Search")
st.logo("Pinecone-Primary-Logo-Black.png")
# Search input
# Quick description of the app, what you can do, and how to use it



st.markdown("""
    ## Welcome to Bird Search!
            
    This app allows you to search for birds using natural language queries across different search methodologies.
            
    You can then log annotations for the search results, based on how many results are relevant (mean average precision) and how many unique relevant birds are returned.
            
    If you'd like to log annotations, please make sure to click on "Log Annotations" after marking relevant results. If a result is not relevant, leave it unchecked.
            
    If all results are irrelevant, hit log annotations to log this fact! 
            
    Want to try some queries? Try these: 
    - "Birds that live in Illinois"
    - "Birds that are bad at flying"
    - "really big birds"
    - "Big bird red head black wings that pecks wood”
    - "Colorful birds that live in the Midwestern United States"
            
    Have fun!
    """)



query = st.text_input("Enter your search query")
# Initialize session state variables for annotations
if 'annotations_df' not in st.session_state:
    st.session_state.annotations_df = pd.DataFrame(columns=[
        'timestamp', 'query', 'method', 'bird', 'rank', 'is_relevant', 'score', 
        'chunk_id', 'chunk_text'
    ])

# Function to handle dataframe updates
def update_annotations_df(edited_df):
    # Ensure all columns have consistent types
    if 'is_relevant' in edited_df.columns:
        edited_df['is_relevant'] = edited_df['is_relevant'].astype(bool)
    if 'rank' in edited_df.columns:
        edited_df['rank'] = edited_df['rank'].astype(int)
    if 'score' in edited_df.columns:
        edited_df['score'] = edited_df['score'].astype(float)
    
    st.session_state.annotations_df = edited_df
    st.success("Annotations updated!")


def calculate_metrics(annotations_df, query=None, method=None):
    """
    Calculate search metrics from annotations
    
    Args:
        annotations_df: DataFrame with annotations
        query: If provided, filter by this query, otherwise use all queries
        method: Search method ("Dense" or "Sparse")
    """
    # Filter by query and method if specified
    filtered_df = annotations_df.copy()
    if query:
        filtered_df = filtered_df[filtered_df['query'] == query]
    if method:
        filtered_df = filtered_df[filtered_df['method'] == method]
    
    # Return zeros if no data after filtering
    if filtered_df.empty:
        return {
            "unique_relevant_birds": 0, 
            "mean_average_precision": 0.0
        }
        
    return {
        "unique_relevant_birds": get_unique_relevant_birds(filtered_df),
        "mean_average_precision": calculate_mean_average_precision(filtered_df)
    }

def visualize_metrics(query=None):
    if st.session_state.annotations_df.empty:
        st.info("No annotations available yet. Mark search results as relevant and log scores.")
        return
    
    # Get unique queries and methods for the selector
    all_queries = st.session_state.annotations_df['query'].unique()
    all_methods = st.session_state.annotations_df['method'].unique()
    
    # Add selector for which query to visualize metrics for
    col1, col2 = st.columns([3, 1])
    with col1:
        metric_query = st.selectbox(
            "Select query to view metrics for:",
            ["All Queries"] + list(all_queries),
            key="metric_query_selector"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        show_details = st.checkbox("Show details", value=True, key="show_metrics_details")
    
    # Determine which query to use
    selected_query = None if metric_query == "All Queries" else metric_query
    
    # Get metrics for all available methods
    method_metrics = {}
    for method in all_methods:
        method_metrics[method] = calculate_metrics(st.session_state.annotations_df, selected_query, method)
    
    # Create metrics DataFrame dynamically based on available methods
    methods_list = []
    metrics_list = []
    values_list = []
    
    for method in all_methods:
        
        # Add relevant birds
        methods_list.append(method)
        metrics_list.append('Relevant Birds')
        values_list.append(int(method_metrics[method]['unique_relevant_birds']))
        
        # Add avg precision
        methods_list.append(method)
        metrics_list.append('Mean Average Precision')
        values_list.append(float(method_metrics[method]['mean_average_precision']))
    
    metrics_data = pd.DataFrame({
        'Method': methods_list,
        'Metric': metrics_list,
        'Value': values_list
    })
    
    # Create bar chart
    title = f"Search Performance Metrics - {metric_query}"
    st.subheader(title)
    
    c = chart.Chart(metrics_data).mark_bar().encode(
        x=chart.X('Method:N'),
        y=chart.Y('Value:Q'),
        color='Method:N',
        column='Metric:N'
    ).properties(width=100)
    
    st.altair_chart(c)
    
    # Show raw metrics in a table if details are requested
    if show_details:
        with st.expander("Detailed Metrics", expanded=True):
            # Create metrics comparison table dynamically
            metrics_comparison_data = {
                'Metric': ['Relevant Birds', 'Mean Average Precision']
            }
            
            for method in all_methods:
                metrics_comparison_data[method] = [
                    int(method_metrics[method]['unique_relevant_birds']),
                    f"{float(method_metrics[method]['mean_average_precision']):.3f}"
                ]
            
            metrics_comparison = pd.DataFrame(metrics_comparison_data)
            st.table(metrics_comparison)
            
            # Show query distribution
            if metric_query == "All Queries":
                st.subheader("Annotations per Query")
                query_counts = st.session_state.annotations_df.groupby(['query', 'method']).size().reset_index(name='count')
                
                # Ensure count column is integer
                query_counts['count'] = query_counts['count'].astype(int)
                
                # Create a grouped bar chart of queries
                query_chart = chart.Chart(query_counts).mark_bar().encode(
                    x=chart.X('query:N', title='Query'),
                    y=chart.Y('count:Q', title='Number of Annotations'),
                    color='method:N',
                    column='method:N'
                ).properties(width=min(80 * len(all_queries), 400))
                
                st.altair_chart(query_chart)



def highlight_matching_words(text: str, query: str) -> str:
    """
    Returns text with words that match the query highlighted in markdown bold.
    
    Args:
        text (str): The original text to process
        query (str): The search query
        
    Returns:
        str: Markdown formatted text with matching words in bold
    """
    # Split query into words and create a set of lowercase words
    query_words = set(query.lower().split())
    
    # Split the text into words, keeping track of original words
    words = text.split()
    
    # Create markdown by bolding matching words
    markdown_words = []
    for word in words:
        # Compare lowercase versions for matching, stripping punctuation
        if word.lower().strip('.,!?()[]{};"\'') in query_words:
            markdown_words.append(f"**{word}**")
        else:
            markdown_words.append(word)
    
    # Join back into text
    return ' '.join(markdown_words)

def display_search_results(results, query, title, container, method):
    with container:
        st.header(title)
        unique_birds = set()
        
        # Add a "Log Annotations" button at the top
        log_key = f"log_{method.lower()}"
        
        # Dictionary to store checkbox states for this method
        if f"{method}_relevance" not in st.session_state:
            st.session_state[f"{method}_relevance"] = {}
        
        for i, hit in enumerate(results):
            bird = hit['fields']['bird']
            text = hit['fields']['chunk_text']
            score = hit['_score']
            chunk_id = hit.get('id', f"{bird}_chunk_{i}")
            
            # Use index and bird name for the checkbox key instead of hit_id
            checkbox_key = f"{method}_{bird}_{i}"
            
            unique_birds.add(bird)
            with st.expander(f"{bird} (Score: {score:.2f})"):
                st.write(highlight_matching_words(text, query))
                
                if bird in parsing_metadata:
                    bird_data = parsing_metadata[bird]
                    if bird_data['images']:
                        image_path = os.path.join("parsed_birds/images", bird_data['images'][0]['local_path'])
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            st.image(image, caption=bird)
                
                # Add relevance checkbox for this result
                st.session_state[f"{method}_relevance"][checkbox_key] = st.checkbox(
                    "Mark as relevant", 
                    key=checkbox_key,
                    value=st.session_state.get(checkbox_key, False)
                )
        
        # Log Annotations button at the bottom
        if st.button(f"Log {method} Annotations", key=log_key):
            # Collect all annotations for this query and method
            new_annotations = []
            for i, hit in enumerate(results):
                bird = hit['fields']['bird']
                text = hit['fields']['chunk_text']
                score = hit['_score']
                chunk_id = hit.get('id', f"{bird}_chunk_{i}")
                
                # Use the same checkbox key format
                checkbox_key = f"{method}_{bird}_{i}"
                is_relevant = st.session_state[f"{method}_relevance"].get(checkbox_key, False)
                
                new_annotations.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'method': method,
                    'bird': bird,
                    'rank': i+1,  # 1-based rank
                    'is_relevant': is_relevant,
                    'score': score,
                    'chunk_id': chunk_id,
                    'chunk_text': text
                })
            
            # Add to annotations dataframe
            if new_annotations:
                new_df = pd.DataFrame(new_annotations)
                
                # Remove any previous annotations for this query and method
                mask = ~((st.session_state.annotations_df['query'] == query) & 
                        (st.session_state.annotations_df['method'] == method))
                st.session_state.annotations_df = pd.concat([
                    st.session_state.annotations_df[mask],
                    new_df
                ]).reset_index(drop=True)
                
                st.success(f"Logged {len(new_annotations)} annotations for {method} search!")
        
        st.subheader(f"Unique Birds Found in {title}")
        st.write(", ".join(unique_birds))
    
    return unique_birds



## Core Mechanisms for App

if query:
    dense_results = query_integrated_inference(query, "dense-bird-search")
    sparse_results = query_integrated_inference(query, "sparse-bird-search")
    bm25_results = query_bm25(query, "bm25-bird-search")
    cascading_results = conduct_cascading_retrieval(query)
    # Tabs for dense vs sparse results
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Keyword Search Results", "Dense Search Results", "Sparse Search Results", "Cascading Retrieval Results", "Metrics & Annotations"])

    with tab1:
        st.write("This method uses BM25 over the entire corpus to retrieve results.")
    with tab2:
        st.write("This method uses a dense embedding model called multilingual-e5-large to retrieve results.")
    with tab3:
        st.write("This method uses sparse retrieval using a proprietary model called pinecone-sparse-english-v0 to retrieve results.")
    with tab4:
        st.write("This method uses cascading retrieval, reranking over the dense and sparse results to retrieve results.")
        st.write("The reranker used is Cohere's Rerank 3.5 Model, which supports reasoning over reranked results.")

    with tab5:
        st.write("This tab shows the metrics and annotations for the search results.")
        st.write("This will calculate two metrics:")
        st.write("1. Mean Average Precision (MAP) - This is a measure of how many relevant results are returned, as a function of the rank of the result.")
        st.write("2. Relevant Birds - This is the number of unique relevant birds in the search results. This is cool, as it tells us how many new birds we learn about!")
        st.write("You can also look at the annotations for each method, and modify, download them as needed.")
        st.write("Try annotating a set of results to see which methods are most effective!")
    unique_bm25_birds = display_search_results(bm25_results, query, "Keyword Search Results", tab1, "Keyword")
    unique_dense_birds = display_search_results(dense_results, query, "Dense Search Results", tab2, "Dense")
    unique_sparse_birds = display_search_results(sparse_results, query, "Sparse Search Results", tab3, "Sparse")
    unique_cascading_birds = display_search_results(cascading_results, query, "Cascading Retrieval Results", tab4, "Cascading")
    # Metrics and annotations tab

    with tab5:
        visualize_metrics()
        
        st.subheader("Annotation History")
        if not st.session_state.annotations_df.empty:
            # Use data_editor instead of dataframe for interactive editing
            st.subheader("Edit Annotations")
            
            # Make a copy to avoid direct reference issues
            edited_df = st.data_editor(
                st.session_state.annotations_df,
                num_rows="dynamic",
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        help="When the annotation was created",
                        format="D MMM YYYY, h:mm a",
                    ),
                    "is_relevant": st.column_config.CheckboxColumn(
                        "Relevant?",
                        help="Check if this result is relevant to the query",
                    ),
                    "score": st.column_config.NumberColumn(
                        "Score",
                        help="Search score",
                        format="%.3f",
                    ),
                    "chunk_text": st.column_config.TextColumn(
                        "Chunk Text",
                        help="The text content of this chunk",
                        width="large",
                    ),
                },
                disabled=["timestamp", "chunk_text"],
                key="annotation_editor",
                on_change=update_annotations_df,
                args=(st.session_state.annotations_df,),
            )
            
            # Update the session state with the edited dataframe
            if not edited_df.equals(st.session_state.annotations_df):
                st.session_state.annotations_df = edited_df
                st.rerun()  # Rerun to update metrics
            
            # Option to download annotations
            csv = st.session_state.annotations_df.to_csv(index=False)
            st.download_button(
                "Download Annotations CSV",
                csv,
                "bird_search_annotations.csv",
                "text/csv",
                key="download-csv"
            )
        else:
            st.info("No annotations logged yet. Use the checkboxes and 'Log Annotations' buttons to evaluate search results.")
