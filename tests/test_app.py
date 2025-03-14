# Contains tests for the annotation feature of the application
from streamlit.testing.v1 import AppTest

def test_annotation_marking_relevant():
    """Test that marking search results as relevant correctly logs annotations"""
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark the first result as relevant in the Dense Search Results tab
    at.tabs[1].checkbox[0].check().run()
    
    # Click the "Log Dense Annotations" button
    at.button(key="log_dense").click().run()
    
    # Verify that the annotation was logged correctly
    assert len(at.session_state["annotations_df"]) == 5  # 5 results from dense search
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 1  # One marked as relevant
    assert at.session_state["annotations_df"]["query"].iloc[0] == query
    assert at.session_state["annotations_df"]["method"].iloc[0] == "Dense"

def test_annotation_marking_none_relevant():
    """Test that logging annotations with no relevant results works correctly"""
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Access the Sparse Search Results tab (index 2) directly
    # Click the "Log Sparse Annotations" button within that tab
    at.tabs[2].button(key="log_sparse").click().run()
    
    # Verify that the annotations were logged correctly
    sparse_results_count = 5  # Expected number of sparse results
    
    assert len(at.session_state["annotations_df"]) == sparse_results_count
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 0  # None marked as relevant
    assert at.session_state["annotations_df"]["query"].iloc[0] == query
    assert at.session_state["annotations_df"]["method"].iloc[0] == "Sparse"

def test_annotation_multiple_results():
    """Test that marking multiple search results as relevant works correctly"""
    # Initialize the app test with a longer timeout
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark first and third results as relevant in the Dense Search Results tab
    at.tabs[1].checkbox[0].check().run()
    at.tabs[1].checkbox[2].check().run()
    
    # Click the "Log Dense Annotations" button
    at.button(key="log_dense").click().run()
    
    # Verify that the annotations were logged correctly
    assert len(at.session_state["annotations_df"]) == 5  # 5 results from dense search
    assert at.session_state["annotations_df"]["is_relevant"].sum() == 2  # Two marked as relevant

def test_metrics_calculation():
    """Test that metrics are calculated correctly based on annotations"""
    # Initialize the app test
    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()
    
    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()
    
    # Mark first result as relevant in Dense search
    at.tabs[1].checkbox[0].check().run()
    at.button(key="log_dense").click().run()
    
    # Now mark first and second results as relevant in Sparse search
    at.tabs[2].checkbox[0].check().run()
    at.tabs[2].checkbox[1].check().run()
    at.tabs[2].button(key="log_sparse").click().run()
    
    # Verify metrics
    dense_annotations = at.session_state["annotations_df"][at.session_state["annotations_df"]["method"] == "Dense"]
    assert dense_annotations["is_relevant"].sum() == 1
    
    sparse_annotations = at.session_state["annotations_df"][at.session_state["annotations_df"]["method"] == "Sparse"]
    assert sparse_annotations["is_relevant"].sum() == 2
    
    all_relevant_birds = at.session_state["annotations_df"][at.session_state["annotations_df"]["is_relevant"]]["bird"].unique()
    assert len(all_relevant_birds) == 3

    # verify MAP and AP metrics are correct for each search method (dense, sparse)

    # access table under last tab
    last_tab = at.tabs[-1]
    table = last_tab.table[0].value

    metrics = table.set_index('Metric').to_dict()
    print(metrics)
    # grab metrics for each method, table is metric rows and methods columns
    assert metrics["Dense"]["Relevant Birds"] == "1", "Dense should have 1 relevant bird"
    assert abs(float(metrics["Dense"]["Mean Average Precision"]) - 1.0) < 0.01, "Dense AP should be 1.0"
    
    # Check Sparse metrics
    assert metrics["Sparse"]["Relevant Birds"] == "2", "Sparse should have 2 relevant birds"
    assert abs(float(metrics["Sparse"]["Mean Average Precision"]) - 1.0) < 0.01, "Sparse AP should be 1.0"
    
    # Check mean_average_precision
    assert abs(float(metrics["Dense"]["Mean Average Precision"]) - 1.0) < 0.01, "MAP should be 1.0"


def test_multiple_queries():
    # annotates the app with two queries, across two methods. Should tell us if MAP is correct

    at = AppTest.from_file("app.py", default_timeout=10)
    at.run()

    # Enter a search query
    query = "red bird"
    at.text_input[0].set_value(query).run()

    # Mark first result as relevant in Dense search
    at.tabs[1].checkbox[0].check().run()
    at.button(key="log_dense").click().run()

    # Mark first result as relevant in Sparse search
    at.tabs[2].checkbox[0].check().run()
    at.tabs[2].button(key="log_sparse").click().run()

    # Enter second query
    query = "blue bird" 
    at.text_input[0].set_value(query).run()

    # Mark first result as relevant in Dense search
    at.tabs[1].checkbox[0].check().run()
    at.button(key="log_dense").click().run()

    # Mark second result as relevant in Sparse search
    at.tabs[2].checkbox[1].check().run()
    at.tabs[2].button(key="log_sparse").click().run()

    # access table under last tab
    last_tab = at.tabs[-1]
    table = last_tab.table[0].value
    metrics = table.set_index('Metric').to_dict()
    print(metrics)
    # grab metrics for each method, table is metric rows and methods columns
    assert metrics["Dense"]["Relevant Birds"] == "2", "Dense should have 2 relevant birds (1 per query)"
    assert abs(float(metrics["Dense"]["Mean Average Precision"]) - 1.0) < 0.01, "Dense MAP should be 1.0 (first result relevant both times)"
    
    # Check Sparse metrics
    assert metrics["Sparse"]["Relevant Birds"] == "2", "Sparse should have 2 relevant birds (1 per query)" 
    assert abs(float(metrics["Sparse"]["Mean Average Precision"]) - 0.75) < 0.01, "Sparse MAP should be 0.75 (1.0 for first query, 0.5 for second query where 2nd result relevant)"

    # Check mean_average_precision matches Dense value
    assert abs(float(metrics["Dense"]["Mean Average Precision"]) - 1.0) < 0.01, "Dense MAP should be 1.0"

