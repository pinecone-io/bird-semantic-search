import nltk
import os

def test_nltk_resources_available():
    """Test that NLTK resources are properly loaded"""
    # Download resources if needed
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Test punkt is available
    try:
        nltk.data.find('tokenizers/punkt')
        punkt_available = True
    except LookupError:
        punkt_available = False
    
    assert punkt_available, "punkt resource not available, please try installing nltk and loading punkt"
    

def test_parsing_metadata_loading():
    """Test that bird metadata loads correctly"""
    import json
    
    # Check if metadata file exists
    metadata_path = "parsed_birds/parsing_metadata.json"
    assert os.path.exists(metadata_path), "Metadata file not found"
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        parsing_metadata = json.load(f)
    
    # Verify it's a dictionary with bird data
    assert isinstance(parsing_metadata, dict)
    assert len(parsing_metadata) > 0, "Metadata is empty, you may need to run the parsing scripts"
    
    # Check structure of a random bird entry
    bird_name = next(iter(parsing_metadata))
    bird_data = parsing_metadata[bird_name]
    assert 'images' in bird_data, "Bird data missing 'images' field"
