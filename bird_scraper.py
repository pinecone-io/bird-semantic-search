import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin
import json

user_agent_string = ""


class WikiBirdScraper:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org"
        self.birds_list_url = "/wiki/List_of_birds_of_North_America"
        self.output_dir = "bird_pages"
        self.headers = {
            'User-Agent': user_agent_string
        }
        self.cache_file = os.path.join(self.output_dir, "scraping_progress.json")
        
    def setup_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def get_bird_links(self):
        """Scrape the main bird list page and return all bird page URLs"""
        response = requests.get(
            urljoin(self.base_url, self.birds_list_url),
            headers=self.headers
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', {'class': 'mw-parser-output'})
        bird_links = []
        
        # Find all h2 headers
        h2_headers = content.find_all('h2')
        
        # Find the range between Tinamous and See also
        start_collecting = False
        for h2 in h2_headers:
            header_text = h2.get_text().strip()
            print(f"Found header: {header_text}")  # Debug print
            
            if 'Tinamous' in header_text:
                start_collecting = True
                print("\nStarting to collect birds...")
            elif 'See also' in header_text:
                print("\nReached See also section, stopping...")
                break
                
            if start_collecting:
                # Get all ul elements until the next h2
                current = h2.find_next()
                while current and current.name != 'h2':
                    if current.name == 'ul':
                        for link in current.find_all('a'):
                            href = link.get('href', '')
                            if (href and 
                                href.startswith('/wiki/') and 
                                ':' not in href and 
                                not href.startswith('/wiki/List_of') and
                                not href.startswith('/wiki/Family_')):
                                
                                bird_links.append({
                                    'href': href,
                                    'name': link.text,
                                    'section': header_text
                                })
                    current = current.find_next()
        
        if not bird_links:
            print("Warning: No birds found! Check if page structure has changed.")
        else:
            total_birds = len(bird_links)
            print(f"\nFound {total_birds} birds in total")
            
            print("\nFirst 10 birds found:")
            for i, bird in enumerate(bird_links[:10], 1):
                print(f"{i}. {bird['name']} ({bird['href']}) - Section: {bird['section']}")
                
            if total_birds > 20:  # Only show last 10 if we have more than 20 birds
                print("\nLast 10 birds found:")
                for i, bird in enumerate(bird_links[-10:], total_birds-9):
                    print(f"{i}. {bird['name']} ({bird['href']}) - Section: {bird['section']}")
            
        return [link['href'] for link in bird_links]
        
    def download_bird_pages(self, limit=None):
        """Download individual bird pages
        Args:
            limit (int, optional): Maximum number of pages to download. Defaults to None (all pages).
        """
        bird_links = self.get_bird_links()
        if limit:
            print(f"\nLimiting to first {limit} birds:")
            for i, link in enumerate(bird_links[:limit]):
                print(f"{i+1}. {link}")
            print()
            bird_links = bird_links[:limit]
        
        # Load existing metadata and downloaded birds
        downloaded_birds = set()
        metadata = {}
        if os.path.exists(os.path.join(self.output_dir, 'metadata.json')):
            try:
                with open(os.path.join(self.output_dir, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                    downloaded_birds = {link for filename, data in metadata.items() 
                                       for link in [data.get('original_link')] if link}
                print(f"Found {len(downloaded_birds)} previously downloaded birds")
            except Exception as e:
                print(f"Error loading existing metadata: {str(e)}")
                
        # Load scraping progress if it exists
        last_processed_idx = -1
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    progress = json.load(f)
                    last_processed_idx = progress.get('last_processed_idx', -1)
                    last_bird = progress.get('last_bird', '')
                    print(f"Resuming from bird #{last_processed_idx + 1} (after {last_bird})")
            except Exception as e:
                print(f"Error loading scraping progress: {str(e)}")
        
        # Filter out already downloaded birds and start from last position
        remaining_links = []
        for idx, link in enumerate(bird_links):
            if idx > last_processed_idx and link not in downloaded_birds:
                remaining_links.append((idx, link))
        
        print(f"Found {len(remaining_links)} new birds to download")
        
        for idx, link in remaining_links:
            try:
                # Construct full URL
                full_url = urljoin(self.base_url, link)
                
                # Create filename from URL
                filename = link.split('/')[-1] + '.html'
                filepath = os.path.join(self.output_dir, filename)
                
                # Download page
                response = requests.get(full_url, headers=self.headers)
                response.raise_for_status()
                
                # Save page content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Store metadata
                metadata[filename] = {
                    'url': full_url,
                    'original_link': link,
                    'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                print(f"Downloaded ({len(metadata)}/{len(bird_links)}): {filename}")
                
                # Update progress
                with open(self.cache_file, 'w') as f:
                    progress = {
                        'last_processed_idx': idx,
                        'last_bird': link
                    }
                    json.dump(progress, f)
                
                # Save metadata after each successful download
                with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Rate limiting - be nice to Wikipedia
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error downloading {link}: {str(e)}")
                # Continue with next bird, don't update progress
                
        # Final save of metadata (redundant but safe)
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    scraper = WikiBirdScraper()
    scraper.setup_directory()
    scraper.download_bird_pages()
if __name__ == "__main__":
    main() 