# Contains tests for the annotation feature of the application
import json
import os
import pandas as pd
from datetime import datetime
import pytest
from streamlit.testing.v1 import AppTest

# Load mock Pinecone results
@pytest.fixture
def mock_pinecone_results():
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "pinecone_results.json")
    with open(fixture_path, 'r') as f:
        return json.load(f)

# Mock the query_integrated_inference and other search functions
def mock_query_functions(monkeypatch, mock_pinecone_results):
    # this lets us just look at the mock data instead of the actual pinecone results
    # to avoid issues with running in actions or elsewhere
    
    def mock_query_integrated_inference(query):
        return mock_pinecone_results[query]["dense"]
    
    
    def mock_query_bm25(query):
        return mock_pinecone_results[query]["bm25"]
    
    def mock_conduct_cascading_retrieval(query):
        return mock_pinecone_results[query]["cascading"]
    
    monkeypatch.setattr("app.query_integrated_inference", mock_query_integrated_inference)
    monkeypatch.setattr("app.query_bm25", mock_query_bm25)
    monkeypatch.setattr("app.conduct_cascading_retrieval", mock_conduct_cascading_retrieval)

def test_annotation_marking_relevant(monkeypatch, mock_pinecone_results):
    """Test that marking search results as relevant correctly logs annotations"""
    # Setup mocks
    mock_query_functions(monkeypatch, mock_pinecone_results)
    
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query that exists in our mock data
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark the first result as relevant in the Dense Search Results tab (index 1)
    # The checkbox key follows the pattern: "Dense_{bird}_{index}"
    first_bird = mock_pinecone_results[query]["dense"][0]["fields"]["bird"]
    checkbox_key = f"Dense_{first_bird}_0"
    at.checkbox(key=checkbox_key).check().run()
    
    # Click the "Log Dense Annotations" button
    at.button(key="log_dense").click().run()
    
    # Verify that the annotation was logged correctly
    # The annotations_df should have one entry with is_relevant=True
    assert len(at.session_state["annotations_df"]) == 5  # 5 results from dense search
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 1  # One marked as relevant
    assert at.session_state["annotations_df"]["query"].iloc[0] == query
    assert at.session_state["annotations_df"]["method"].iloc[0] == "Dense"
    assert at.session_state["annotations_df"]["bird"].iloc[0] == first_bird

def test_annotation_marking_none_relevant(monkeypatch, mock_pinecone_results):
    """Test that logging annotations with no relevant results works correctly"""
    # Setup mocks
    mock_query_functions(monkeypatch, mock_pinecone_results)
    
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query that exists in our mock data
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Access the Sparse Search Results tab (index 2) directly
    # Click the "Log Sparse Annotations" button within that tab
    at.tabs[2].button(key="log_sparse").click().run()
   
    
    # Verify that the annotations were logged correctly
    # The annotations_df should have entries with is_relevant=False for all results
    sparse_results = mock_pinecone_results[query]["sparse"]
    expected_count = len(sparse_results)
    
    assert len(at.session_state["annotations_df"]) == expected_count
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 0  # None marked as relevant
    assert at.session_state["annotations_df"]["query"].iloc[0] == query
    assert at.session_state["annotations_df"]["method"].iloc[0] == "Sparse"


## mock a bunch of annotations, and test that the metrics are correct

def test_annotation_multiple_results(monkeypatch, mock_pinecone_results):
    """Test that marking multiple search results as relevant works correctly"""
    # Setup mocks
    mock_query_functions(monkeypatch, mock_pinecone_results)
    
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query that exists in our mock data
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark multiple results as relevant in the Dense Search Results tab
    dense_results = mock_pinecone_results[query]["dense"]
    
    # Mark first and third results as relevant
    first_bird = dense_results[0]["fields"]["bird"]
    third_bird = dense_results[2]["fields"]["bird"]
    
    at.checkbox(key=f"Dense_{first_bird}_0").check().run()
    at.checkbox(key=f"Dense_{third_bird}_2").check().run()
    
    # Click the "Log Dense Annotations" button
    at.button(key="log_dense").click().run()
    
    # Verify that the annotations were logged correctly
    assert len(at.session_state["annotations_df"]) == 5  # 5 results from dense search
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 2  # Two marked as relevant
    
    # Check that the correct birds were marked
    relevant_birds = at.session_state["annotations_df"][at.session_state["annotations_df"]["is_relevant"]]["bird"].tolist()
    assert first_bird in relevant_birds
    assert third_bird in relevant_birds

def test_metrics_calculation(monkeypatch, mock_pinecone_results):
    """Test that metrics are calculated correctly based on annotations"""
    # Setup mocks
    mock_query_functions(monkeypatch, mock_pinecone_results)
    
    # Initialize the app test
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark first result as relevant in Dense search
    dense_results = mock_pinecone_results[query]["dense"]
    first_dense_bird = dense_results[0]["fields"]["bird"]
    at.checkbox(key=f"Dense_{first_dense_bird}_0").check().run()
    at.button(key="log_dense").click().run()
    
    # Now mark first and second results as relevant in Sparse search
    sparse_results = mock_pinecone_results[query]["sparse"]
    first_sparse_bird = sparse_results[0]["fields"]["bird"]
    second_sparse_bird = sparse_results[1]["fields"]["bird"]
    at.checkbox(key=f"Sparse_{first_sparse_bird}_0").check().run()
    at.checkbox(key=f"Sparse_{second_sparse_bird}_1").check().run()
    at.button(key="log_sparse").click().run()
        
    # Verify metrics for Dense search
    # For Dense: 1 relevant out of 5, at position 1 -> MAP = 1.0, Unique birds = 1
    dense_annotations = at.session_state["annotations_df"][at.session_state["annotations_df"]["method"] == "Dense"]
    assert dense_annotations["is_relevant"].sum() == 1
    
    # For Sparse: 2 relevant out of 5, at positions 1 and 2 -> MAP = (1/1 + 2/2)/2 = 1.0, Unique birds = 2
    sparse_annotations = at.session_state["annotations_df"][at.session_state["annotations_df"]["method"] == "Sparse"]
    assert sparse_annotations["is_relevant"].sum() == 2
    
    # Total unique relevant birds should be 3 if all birds are different
    all_relevant_birds = at.session_state["annotations_df"][at.session_state["annotations_df"]["is_relevant"]]["bird"].unique()
    assert len(all_relevant_birds) == 3

