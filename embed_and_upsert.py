from pinecone import Pinecone, SparseValues, Vector, ServerlessSpec
import re
from tqdm import tqdm
# Read in Text Data
import json
from pinecone_text.sparse import BM25Encoder
import os
import streamlit as st

pc = Pinecone(api_key=st.secrets["pinecone_api_key"])

dense_index_name = "dense-bird-search"
sparse_index_name = "sparse-bird-search"
bm25_index_name = "bm25-bird-search"
bm25_file_name = "bm25_birds.json"


if not pc.has_index(dense_index_name):
    index_model = pc.create_index_for_model(
        name=dense_index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"multilingual-e5-large",
            "field_map":{"text": "chunk_text"}
        }
    )

if not pc.has_index(sparse_index_name):
    index_model = pc.create_index_for_model(
        name=sparse_index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"pinecone-sparse-english-v0",
            "field_map":{"text": "chunk_text"}
        }
    )

if not pc.has_index(bm25_index_name):
    index_model = pc.create_index(
        name=bm25_index_name,
        metric="dotproduct",
        vector_type="sparse",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        )
    )




# Chunk Text Data by Bird, providing unique chunk id and docid

def chunk_text(text, docid, chunk_size=1000):
    '''splits on newlines, then subsplits if larger than chunk_size
    '''

    #remove .txt from docid
    docid = docid.replace(".txt", "")
    chunks = []
    # Split on newlines
    text_lines = text.split('\n')
    for line in (text_lines):
        if len(line) > chunk_size:
            # Split on sentence basis
            sentences = re.split(r'(?<=[.!?])', line)
            for sentence in sentences:
                chunks.append(sentence)
        else:
            chunks.append(line)
    # Clean list comprehension for docis, chunkid
    # TODO: better condition on filtering out trivially small chunks. Maybe we check how many words there are, and adjust for that?
    
    chunks = [chunk for chunk in chunks if len(chunk) > 5]

    chunks = [ {"chunkid": f"doc{docid}#chunk{chunk_num}", "chunk_text": chunk} for chunk_num, chunk in enumerate(chunks)]
    return chunks

def process_text_data(text_data, parsing_metadata):
    '''Processes text data by bird, providing unique docid and chunkid.
    Returns list of chunks in record format for Pinecone upsert
    
    '''
    records = []
    for bird, metadata in parsing_metadata.items():
        text = text_data[metadata["text_file"]]
        chunks = chunk_text(text, metadata["text_file"])
        # filter out chunks with 0 length

        for chunk in chunks:
            records.append(
                {
                    "_id": chunk["chunkid"],
                    "chunk_text": chunk["chunk_text"],
                    "bird": bird,
                }
            )
    return records


# embed/upsert to Pinecone in batches (sparse, dense)
def batched_embed_and_upsert(records, index_name, namespace, batch_size=96):
    '''Upserts records to Pinecone in batches of batch_size. We are limited by the embedding model AND the upsert limits here'''
    index = pc.Index(index_name)

    if index_name == "dense-bird-search":
        # only works for dense indexes right now
        existing_ids_iterator = index.list(namespace=namespace, limit=100)
        existing_ids = []
        for id_batch in tqdm(existing_ids_iterator, desc="Checking for existing IDs in Dense Index"):
            existing_ids.extend(id_batch)
        
        # filter records to just include records who's _id is not in existing_ids
        records = [record for record in records if record["_id"] not in existing_ids]

    for i in tqdm(range(0, len(records), batch_size), 
                  desc="Upserting records to Pinecone"):
        batch = records[i:i+batch_size]
        try:
            index.upsert_records(records=batch, namespace=namespace)
        except Exception as e:
            print(f"Error upserting batch: {e}")
            print(f"Batch: {batch}")


def bm25_batch_encode_upsert(records, index_name, namespace, batch_size=100):
    '''Upserts records using BM25 to Pinecone in batches of batch_size 
    '''

    index = pc.Index(index_name)

    # Initialize BM25 and fit the corpus.
    bm25 = BM25Encoder()
    all_text = [record["chunk_text"] for record in records]

    # "train" the BM25 encoder
    print("Fitting BM25 encoder")
    bm25.fit(all_text)

    bm25.dump(bm25_file_name)

    vectors = []
    print("Encoding BM25 Records")
    encoded_corpus = bm25.encode_documents(all_text)
    #check encoded corpus for any empty list values

    #empty vectors
    empty_vectors = []
    for i, e in enumerate(encoded_corpus):
        if len(e["values"]) == 0:
            empty_vectors.append({"index": i, "record": records[i]})
    print(f"Found {len(empty_vectors)} empty vectors for BM25. Removing them...")

    for i in empty_vectors:
        print(f"Empty Vector at index {i['index']}")
        print(f"Record: {i['record']}")
    

    for r, e in tqdm(zip(records, encoded_corpus), desc="Transforming BM25 Records"):

        new_vector = Vector(
            id=r["_id"],
            sparse_values=SparseValues(
                values=e["values"],
                indices=e["indices"]
            ),
            metadata={
                "chunk_text": r["chunk_text"],
                "bird": r["bird"]
            }
        )
        vectors.append(new_vector)

    # Remove at this step, to avoid weird index issues
    vectors = list(filter(lambda x: len(x.sparse_values.values) > 0, vectors))
    
    for i in tqdm(range(0, len(vectors), batch_size), 
                  desc="Upserting BM25 Records"):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            #print(batch)
            print(f"Error upserting batch: {e}")
            break





if __name__ == "__main__":
    with open("parsed_birds/parsing_metadata.json", 'r') as f:
        parsing_metadata = json.load(f)

    # Change to dictionary instead of list
    text_data = {}

    for bird, metadata in parsing_metadata.items():
        text_file = metadata["text_file"]
        with open(os.path.join("parsed_birds/text", text_file), 'r', encoding='utf-8') as f:
            text_data[text_file] = f.read()  # Store with filename as key
    
    records = process_text_data(text_data, parsing_metadata)
    print("Found", len(records), "records to upsert")
    
    # Upsert Dense Embeddings, might take a few minutes
    print("Upserting Dense Embeddings")
    #batched_embed_and_upsert(records, dense_index_name, "bird-search")
    
    # Upsert Sparse Embeddings, takes a few minutes
    print("Upserting Sparse Embeddings")
    batched_embed_and_upsert(records, sparse_index_name, "bird-search")

    # Upsert BM25 Embeddings
    print("Upserting BM25 Embeddings")
    #bm25_batch_encode_upsert(records, bm25_index_name, "bird-search")




