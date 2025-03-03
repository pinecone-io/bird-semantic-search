from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os 
import streamlit as st

# Initialize Pinecone and key parameters
TOP_K = 5
bm25_file_name = "bm25_birds.json"
pc = Pinecone(api_key=st.secrets["pinecone_api_key"])


def query_integrated_inference(query, index_name, namespace="bird-search"):
    index = pc.Index(index_name)
    results = index.search_records(
        namespace=namespace,
        query={
            "inputs": {"text": query},
            "top_k": TOP_K,
        },

        fields=["bird", "chunk_text"] # Request specific fields from records
    )
    return results['result']['hits'] # Access hits from result object

def query_rerank_integrated_inference(query, index_name, namespace="bird-search"):
    index=pc.Index(index_name)
    sr = index.search(
        namespace=namespace,
        query={
                "top_k": TOP_K * 2,
                "inputs": {
                    "text": query
                }
        },
        rerank={
            "model": "cohere-rerank-3.5",
            "rank_fields": ["chunk_text"]
        }
    )
    return sr['result']['hits']
def query_bm25(query, index_name, namespace="bird-search"):

    index = pc.Index(index_name)

    bm25 = BM25Encoder().load(path=bm25_file_name)
    encoded_query = bm25.encode_queries(query)
    # query the db

    results = index.query(
        namespace=namespace,
        sparse_vector={
            "values": encoded_query["values"],
            "indices": encoded_query["indices"]
        },
        top_k=TOP_K,
        include_metadata=True
    )
    # Different format than for integrated inference!

    # reformat into the same format as the integrated inference
    final_results = []
    for r in results['matches']:
        final_results.append({
            'id': r['id'],
            'fields': r['metadata'],
            '_score': r['score'],
        })

    return final_results

def conduct_cascading_retrieval(query, sparse_index_name="sparse-bird-search", dense_index_name="dense-bird-search", namespace="bird-search"):
    '''Conduct cascading retrieval, reranking across sparse and dense indexes. Returns TOP_K results.'''
    # Conduct dense retrieval
    dense_results = query_rerank_integrated_inference(query, dense_index_name, namespace)

    # Conduct sparse retrieval
    sparse_results = query_rerank_integrated_inference(query, sparse_index_name, namespace)

    # combine results into one list
    combined_results = dense_results + sparse_results

    # dedup results on chunk_id
    # Create a dictionary to track seen IDs
    seen_ids = {}
    deduped_results = []
    
    # Keep first occurrence of each ID
    for result in combined_results:
        if result['_id'] not in seen_ids:
            seen_ids[result['_id']] = True
            deduped_results.append(result)
    
    combined_results = deduped_results

    # sort results by score, highest to lowest
    combined_results = sorted(combined_results, key=lambda x: x['_score'], reverse=True)

    # return TOPK results
    return combined_results[:TOP_K]
