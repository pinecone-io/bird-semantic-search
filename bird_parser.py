import os
import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
from pathlib import Path
import time

user_agent_string = ""

class WikiBirdParser:
    def __init__(self):
        self.pages_dir = "bird_pages"
        self.output_dir = "parsed_birds"
        self.text_dir = os.path.join(self.output_dir, "text")
        self.images_base_dir = os.path.join(self.output_dir, "images")
        self.parsed_metadata = {}
        self.headers = {
            'User-Agent': user_agent_string
        }
        self.cache_file = os.path.join(self.output_dir, "parsing_progress.json")
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.output_dir, self.text_dir, self.images_base_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
                
    def clean_text(self, text):
        """Clean extracted text by removing extra whitespace and citations"""
        # Remove citation brackets [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def is_valid_bird_image(self, image, image_url, alt_text):
        """Check if the image is a valid bird image"""
        if not image or not image_url:
            return False
            
        # Skip status/range maps and icons
        if ('status' in image_url.lower() or 
            'range' in image_url.lower() or 
            'icon' in image_url.lower() or 
            'map' in image_url.lower()):
            return False
            
        # Skip small icons/decorative images
        width = int(image.get('width', 0))
        if width and width < 100:
            return False
            
        return True
    
    def categorize_image(self, alt_text, caption_text=""):
        """Categorize image based on alt text and caption"""
        text = (alt_text + " " + caption_text).lower()
        
        # Look for gender indicators
        if any(term in text for term in ['male', 'female']):
            if 'male' in text and 'female' in text:
                return 'both_genders'
            elif 'male' in text:
                return 'male'
            else:
                return 'female'
        
        return 'unspecified'
    
    def download_image(self, image_url, image_path):
        """Download an image and return success status"""
        try:
            img_response = requests.get(image_url, headers=self.headers)
            img_response.raise_for_status()
            
            with open(image_path, 'wb') as f:
                f.write(img_response.content)
            return True
        except Exception as e:
            print(f"Error downloading image {image_url}: {str(e)}")
            return False
                
    def parse_bird_page(self, filename):
        """Parse a single bird page for text and image"""
        bird_name = filename.replace('.html', '')
        
        # Skip if already parsed
        if bird_name in self.parsed_metadata:
            print(f"Bird {bird_name} already parsed. Skipping.")
            return True
            
        with open(os.path.join(self.pages_dir, filename), 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Create bird-specific image directory
        bird_image_dir = os.path.join(self.images_base_dir, bird_name)
        Path(bird_image_dir).mkdir(exist_ok=True)
        
        # Extract main content
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            print(f"No content found for {bird_name}")
            return False
            
        # Get paragraphs
        paragraphs = []
        for p in content.find_all('p'):
            if p.text.strip():
                cleaned_text = self.clean_text(p.text)
                if cleaned_text:
                    paragraphs.append(cleaned_text)
        
        # Save text content
        text_filename = f"{bird_name}.txt"
        text_path = os.path.join(self.text_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(paragraphs))
            
        # Find and download images - focusing only on infobox
        images_info = []
        infobox = soup.find('table', {'class': 'infobox'})
        
        if infobox:
            for img in infobox.find_all('img')[:2]:  # Only look at first two images
                # Get caption if available
                caption = ""
                caption_elem = img.find_parent('td').find_next_sibling('td') if img.find_parent('td') else None
                if caption_elem:
                    caption = caption_elem.get_text(strip=True)
                
                image_url = urljoin('https:', img.get('src', ''))
                alt_text = img.get('alt', '')
                
                if self.is_valid_bird_image(img, image_url, alt_text):
                    image_info = {
                        'url': image_url,
                        'alt': alt_text,
                        'caption': caption,
                        'width': img.get('width', 'unknown'),
                        'height': img.get('height', 'unknown'),
                        'type': self.categorize_image(alt_text, caption)
                    }
                    images_info.append(image_info)
                    
                    if len(images_info) >= 2:  # Stop after finding 2 valid images
                        break
                
        # Download and save images
        successful_images = []
        for i, image_info in enumerate(images_info):
            # Create descriptive filename based on type
            image_type = image_info['type']
            if image_type in ['male', 'female']:
                image_filename = f"{bird_name}_{image_type}.jpg"
            else:
                image_filename = f"{bird_name}_{i+1}.jpg"
            
            image_path = os.path.join(bird_image_dir, image_filename)
            
            # Check if image already exists
            if os.path.exists(image_path):
                print(f"Image {image_filename} already exists for {bird_name}")
                image_info['local_path'] = os.path.join(bird_name, image_filename)
                successful_images.append(image_info)
                continue
                
            if self.download_image(image_info['url'], image_path):
                image_info['local_path'] = os.path.join(bird_name, image_filename)
                successful_images.append(image_info)
        
        # Store metadata
        self.parsed_metadata[bird_name] = {
            'text_file': text_filename,
            'paragraphs': len(paragraphs),
            'images': successful_images,
            'total_images': len(successful_images),
            'has_male_female_pairs': any(img['type'] == 'male' for img in successful_images) and 
                                   any(img['type'] == 'female' for img in successful_images)
        }
        
        # Save parsing metadata after each bird to allow resuming
        with open(os.path.join(self.output_dir, 'parsing_metadata.json'), 'w') as f:
            json.dump(self.parsed_metadata, f, indent=2)
            
        # Update progress cache
        with open(self.cache_file, 'w') as f:
            json.dump({
                'last_parsed_bird': bird_name,
                'total_parsed': len(self.parsed_metadata),
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f)
            
        return True
        
    def parse_all_birds(self):
        """Parse all downloaded bird pages"""
        self.setup_directories()
        
        # Load original metadata to get list of files
        with open(os.path.join(self.pages_dir, 'metadata.json'), 'r') as f:
            original_metadata = json.load(f)
            
        # Load existing parsing metadata if available
        if os.path.exists(os.path.join(self.output_dir, 'parsing_metadata.json')):
            with open(os.path.join(self.output_dir, 'parsing_metadata.json'), 'r') as f:
                self.parsed_metadata = json.load(f)
                print(f"Loaded existing metadata for {len(self.parsed_metadata)} birds")
                
        # Load parsing progress if available
        last_parsed_bird = None
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                progress = json.load(f)
                last_parsed_bird = progress.get('last_parsed_bird')
                print(f"Last parsed bird: {last_parsed_bird}")
                
        # Get all HTML files to parse
        files_to_parse = [f for f in original_metadata.keys() if f.endswith('.html')]
        total_files = len(files_to_parse)
        print(f"Found {total_files} bird pages to parse")
        
        # Find starting point if resuming
        start_idx = 0
        if last_parsed_bird:
            last_filename = f"{last_parsed_bird}.html"
            if last_filename in files_to_parse:
                start_idx = files_to_parse.index(last_filename) + 1
                print(f"Resuming from file #{start_idx} (after {last_parsed_bird})")
                
        # Parse each file
        for i, filename in enumerate(files_to_parse[start_idx:], start_idx):
            print(f"Parsing ({i+1}/{total_files}): {filename}")
            success = self.parse_bird_page(filename)
            if not success:
                print(f"Failed to parse {filename}, continuing with next file")
                
        # Final save of parsing metadata
        with open(os.path.join(self.output_dir, 'parsing_metadata.json'), 'w') as f:
            json.dump(self.parsed_metadata, f, indent=2)
            
def main():
    parser = WikiBirdParser()
    parser.parse_all_birds()

if __name__ == "__main__":
    main() 