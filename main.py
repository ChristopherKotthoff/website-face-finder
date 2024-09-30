import argparse
import tempfile
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import cv2
import numpy as np
import requests
import time
from tqdm import tqdm

from helpers import Recognizer, WebScraper, draw_bounding_box


def get_image_from_url(url):
    try:
        # Send a GET request to the image URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful

        # Convert the response content into a numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # Decode the image array to get an image in a format that OpenCV understands
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve image from {url}: {e}")
        return None

def argument_handling():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Search for similar faces on the web.")

    # Add arguments
    parser.add_argument("--headless", action="store_true", help="Interactions are via command line only. --input is required.", required=False)
    parser.add_argument("--restore", action="store_true", help="Restore state from the state.txt file", required=False)
    parser.add_argument("--input", type=str, help="Face image to look for", required=False)
    parser.add_argument("--output", type=str, help="Output folder", default=tempfile.gettempdir(), required=False)
    parser.add_argument("--domain", type=str, help="The domain to be considered", required=False)

    # Parse the arguments
    args = parser.parse_args()

    # Check if output directory exists
    if not os.path.exists(args.output):
        print("The output directory does not exist. Exiting...")
        exit(1)

    root = None
    if not args.headless:
        root = tk.Tk()
        root.withdraw()

        # Use ttk's modern theme
        style = ttk.Style()
        style.theme_use("clam")

    # input
    if args.input is None:
        if args.headless:
            args.input = input("Enter the path to the input file: ")
        else:
            file_path = filedialog.askopenfilename(
                title="Select an image file",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")]
            )
            if not file_path:
                print("No file selected. Exiting...")
                exit(1)
            args.input = file_path

    # Check if input is valid
    if not os.path.exists(args.input):
        print("The input file does not exist. Exiting...")
        exit(1)

    # domain
    if args.domain is None:
        if args.headless:
            args.domain = input("Enter the domain to be considered: ")
        else:
            # Open modern-looking input dialog for domain input
            user_input = simpledialog.askstring("Input", "Enter the domain to be considered")
            if not user_input:
                print("No domain entered. Exiting...")
                exit(1)
            args.domain = user_input

    # Check if domain is valid
    if len(args.domain.split(".")) < 2 or not args.domain.startswith("http"):
        print("Invalid domain. Should be like <https://www.example.com> or <https://subexample.example.com> or similar. Exiting...")
        exit(1)

    # Create output directory for current run, if exists, append number higher
    output_dir = args.output
    if os.path.exists(output_dir):
        i = 1
        while os.path.exists(output_dir):
            output_dir = os.path.join(args.output, f"output_{i}")
            i += 1

    # Create output directory
    os.makedirs(output_dir)

    return args.headless, args.input, output_dir, args.domain, args.restore

if __name__ == "__main__":
    headless, input_file, output_dir, domain, restore = argument_handling()

    print(f"Headless mode: \t\t{headless}")
    print(f"Input file: \t\t{input_file}")
    print(f"Output directory: \t{output_dir}")
    print(f"Domain: \t\t{domain}")
    
    recognizer = Recognizer()
    scraper = WebScraper(domain)
    if restore:
        scraper.load_state()
    
    face_embedding = recognizer.get_faces(cv2.imread(input_file))[0].embedding
    
    consecutive_errors = 0
    
    iter = 0
    while True:
        iter += 1
        if iter % 10 == 0:
            scraper.save_state()
        try:
            if scraper.len(scraper.image_queue) < 1000:
                new_imgs, new_links = scraper.search(20)
                print(f"Found {new_imgs} new images and {new_links} new links.")
            for _ in tqdm(range(len(scraper.image_queue))):
                img_url, img_found_url = scraper.get_next_image_info()
                
                #tqdm.write(f"Next image URL: {img_url}")
                
                img = get_image_from_url(img_url)
                
                faces_strangers = recognizer.get_faces(img)
                
                if not faces_strangers:
                    continue
                
                max_similarity = 0
                
                for stranger_face in faces_strangers:
                    similarity = recognizer.cosine_similarity(face_embedding, stranger_face.embedding)
                    max_similarity = max(max_similarity, similarity)
                    draw_bounding_box(img, stranger_face, similarity)
                
                cv2.imwrite(f'outputs/{img_url.replace("/",".")}', img)
                if max_similarity > 0.3:
                    print(f"Found a promising face with similarity {max_similarity:.2f} in {img_found_url}.")
                    cv2.imwrite(f'outputs/promising/{img_url.replace("/",".")}', img)
                    # append to promising_faces.txt
                    with open('promising_faces.txt', 'a') as f:
                        f.write(f"{max_similarity:.2f} {img_url} {img_found_url} \n")
            
            consecutive_errors = 0
                    
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1.2**consecutive_errors)
            consecutive_errors += 1



