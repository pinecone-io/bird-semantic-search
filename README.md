# Bird Semantic Search

A semantic search application on a corpus of North American Bird data from Wikipedia. This project demonstrates different search methodologies and allows users to evaluate them using Streamlit and Pinecone.

# Bird Search App

## Overview

This application allows users to search for information about North American birds using natural language queries. It implements and compares multiple search methodologies:

- **Dense Vector Search**: Using multilingual-e5-large embeddings to capture semantic meaning
- **Sparse Vector Search**: Using Pinecone's sparse-english-v0 model for keyword-focused retrieval
- **BM25 Search**: Traditional keyword-based search using the BM25 algorithm
- **Cascading Retrieval**: A hybrid approach that combines dense and sparse search with reranking

## Features

- **Natural Language Queries**: Search for birds using everyday language
- **Multiple Search Methods**: Compare different search technologies side-by-side
- **Interactive Results**: Expand/collapse search results with bird images
- **Relevance Annotation**: Mark results as relevant/irrelevant for evaluation
- **Performance Metrics**: Calculate and visualize search quality metrics
  - Mean Average Precision (MAP)
  - Unique Relevant Birds
- **Example Queries**: Pre-defined example queries to explore the system
- **Data Export**: Download annotations for further analysis

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bird-semantic-search.git
   cd bird-semantic-search
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Pinecone API key:
   ```
   export PINECONE_API_KEY=your_api_key
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter a natural language query or select one of the example queries
2. View results from different search methods in the tabs
3. Expand results to see detailed information and bird images
4. Mark relevant results using the checkboxes
5. Click "Log Annotations" to record your relevance judgments
6. Switch to the Metrics tab to see performance comparisons
7. Download your annotations for further analysis

## Search Methodologies

### Dense Vector Search
Uses the multilingual-e5-large model to convert queries and documents into dense vector embeddings. This method excels at capturing semantic meaning and can find relevant results even when they don't share exact keywords with the query.

### Sparse Vector Search
Leverages Pinecone's sparse-english-v0 model to create sparse vector representations that focus on important keywords. This approach combines the efficiency of traditional keyword search with some semantic understanding.

### BM25 Search
Implements the BM25 algorithm, a traditional information retrieval method that ranks documents based on term frequency and inverse document frequency. This serves as a baseline for comparison.

### Cascading Retrieval
A hybrid approach that first performs dense search, then adds sparse search results for birds not already found, and finally reranks everything using Cohere's Rerank 3.5 model. This method aims to combine the strengths of multiple approaches.

## Evaluation Metrics

### Mean Average Precision (MAP)
Measures how many relevant results are returned as a function of their rank. This metric rewards search methods that place relevant results higher in the result list.

### Unique Relevant Birds
Counts the number of unique bird species found that are relevant to the query. This metric helps evaluate the diversity of relevant results.

## Data Source

The bird information is sourced from Wikipedia articles about North American birds, processed and chunked for efficient retrieval.

