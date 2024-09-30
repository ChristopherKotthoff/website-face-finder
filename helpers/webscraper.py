import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.registered_urls = set()
        self.images_urls = set()
        self.image_queue = deque()
        self.urls_to_visit = deque([base_url])
        self.domain = urlparse(base_url).netloc
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def is_valid_image(self, img_url):
        return img_url.lower().endswith(('jpg', 'jpeg', 'png'))

    def get_absolute_url(self, link):
        return urljoin(self.base_url, link)

    def is_internal_link(self, link):
        return urlparse(link).netloc == self.domain or urlparse(link).netloc == ''

    def scrape_page(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Ensure the request was successful
        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return 0, 0
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        new_imgs = 0
        new_links = 0

        # Find all images and put valid image URLs into the image queue
        images = soup.find_all('img')
        for img in images:
            img_url = img.get('src')
            if img_url:
                absolute_img_url = self.get_absolute_url(img_url)
                if self.is_valid_image(absolute_img_url):
                    with self.lock:  # Lock to ensure thread-safe access to image data
                        if absolute_img_url not in self.images_urls:
                            new_imgs += 1
                            self.images_urls.add(absolute_img_url)
                            self.image_queue.append((absolute_img_url, url))
        
        # Find all links on the page and add them to the urls_to_visit if internal and unvisited
        links = soup.find_all('a', href=True)
        for link in links:
            href = link.get('href')
            absolute_url = self.get_absolute_url(href)
            if self.is_internal_link(absolute_url) and not absolute_url.startswith(("mailto:", "tel:")):
                with self.lock:  # Lock to ensure thread-safe access to URL data
                    if absolute_url not in self.registered_urls and not " " in absolute_url:
                        new_links += 1
                        self.registered_urls.add(absolute_url)
                        self.urls_to_visit.append(absolute_url)
        
        return new_imgs, new_links
    
    def save_state(self):
        with self.lock:  # Lock to ensure thread-safe access during state saving
            with open('state.txt', 'w') as f:
                f.write(f"{self.base_url}\n")
                f.write(f"{self.domain}\n")
                f.write(f"{' '.join(self.registered_urls)}\n")
                f.write(f"{' '.join(self.images_urls)}\n")
                f.write(f"{' '.join(self.urls_to_visit)}\n")
                f.write(f"{' '.join([f'{img_url} {page_url}' for img_url, page_url in self.image_queue])}\n")
            
    def load_state(self):
        with self.lock:  # Lock to ensure thread-safe access during state loading
            with open('state.txt', 'r') as f:
                self.base_url = f.readline().strip()
                self.domain = f.readline().strip()
                self.registered_urls = set(f.readline().strip().split())
                self.images_urls = set(f.readline().strip().split())
                self.urls_to_visit = deque(f.readline().strip().split())
                self.image_queue = deque([tuple(line.split()) for line in f.readline().strip().split()])

    def search(self, concurrent=1):
        if not self.urls_to_visit:
            print("No more URLs to visit.")
            return

        new_imgs = 0
        new_links = 0
        # Using ThreadPoolExecutor for concurrent URL scraping
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = []
            for _ in range(min(concurrent, len(self.urls_to_visit))):
                next_url = self.urls_to_visit.popleft()  # Get the next URL to scrape
                #print(f"Scraping {next_url}")
                futures.append(executor.submit(self.scrape_page, next_url))
            
            
            for future in as_completed(futures):
                result = future.result()  # Wait for the future to complete
                if result:
                    new_imgs += result[0]
                    new_links += result[1]
                    #print(f"Scraped {result[0]} new images and {result[1]} new links.")
            return new_imgs, new_links

    def get_next_image_info(self):
        with self.lock:  # Lock to ensure thread-safe access to image queue
            if self.image_queue:
                return self.image_queue.popleft()
            else:
                return None


if __name__ == '__main__':
    # Example Usage
    scraper = WebScraper("https://www.persoenlich.com")

    # To search and scrape for multiple pages
    while True:
        scraper.search(10)
        while scraper.image_queue:
            print("Next image URL:", scraper.get_next_image_info())
        print("waiting for user input to continue...")
        input()
