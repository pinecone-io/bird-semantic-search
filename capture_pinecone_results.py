"""
Script to capture real Pinecone results for testing.
Run this manually to update the test fixtures.
"""
import json
import os

from query_db import query_bm25, query_integrated_inference, conduct_cascading_retrieval

# Ensure output directory exists
os.makedirs("tests/fixtures", exist_ok=True)

# Test queries to capture results for
TEST_QUERIES = [
    "red bird",
    "blue bird with long tail",
    "small yellow bird that sings",
    "birds in Illinois"
]

def extract_simplified_result(result):
    """Extract only the fields we need from a result object or dictionary"""
    if isinstance(result, dict):
        # Extract fields from dictionary
        return {
            "id": result.get("id", ""),
            "_id": result.get("_id", ""),
            "_score": result.get("_score", 0.0),
            "score": result.get("score", 0.0),
            "fields": {
                "bird": result.get("fields", {}).get("bird", ""),
                "chunk_text": result.get("fields", {}).get("chunk_text", "")
            }
        }
    else:
        # Extract fields from object
        fields = getattr(result, "fields", {}) or {}
        return {
            "id": getattr(result, "id", ""),
            "_id": getattr(result, "_id", ""),
            "_score": getattr(result, "_score", 0.0),
            "score": getattr(result, "score", 0.0),
            "fields": {
                "bird": fields.get("bird", ""),
                "chunk_text": fields.get("chunk_text", "")
            }
        }

def process_results(results):
    """Process a list of results into simplified format"""
    simplified_results = []
    for result in results:
        simplified_results.append(extract_simplified_result(result))
    return simplified_results

def capture_results():
    """Capture results from all search methods for test queries"""
    # Create a dictionary to store all results
    all_results = {}
    
    for query in TEST_QUERIES:
        print(f"\nProcessing query: {query}")
        
        # Dictionary to store results for this query
        query_results = {}
        
        # Get and process results for each method
        methods = {
            "bm25": lambda: query_bm25(query, "bm25-bird-search"),
            "dense": lambda: query_integrated_inference(query, "dense-bird-search"),
            "sparse": lambda: query_integrated_inference(query, "sparse-bird-search"),
            "cascading": lambda: conduct_cascading_retrieval(query)
        }
        
        for method_name, method_func in methods.items():
            print(f"Getting {method_name} results...")
            try:
                results = method_func()
                print(f"Found {len(results)} {method_name} results")
                query_results[method_name] = process_results(results)
                # print the results
                print(f"Results for {query}: {query_results[method_name]}")
            except Exception as e:
                print(f"Error getting {method_name} results: {e}")
                query_results[method_name] = []
        
        # Store results for this query
        all_results[query] = query_results
    
    # Save results to file
    output_path = "tests/fixtures/pinecone_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Verify saved results
    with open(output_path, "r") as f:
        saved_data = json.load(f)
    
    print("\nVerification of saved results:")
    for query in TEST_QUERIES:
        for method in ["bm25", "dense", "sparse", "cascading"]:
            result_count = len(saved_data[query][method])
            print(f"  {query} - {method}: {result_count} results")

if __name__ == "__main__":
    capture_results()