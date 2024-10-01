import argparse
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
        response = requests.get(url, timeout=3)
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
    parser.add_argument("--input", type=str, help="Face image of person to look for", required=False)
    parser.add_argument("--output", type=str, help="Location where the output folder is created", default="./", required=False)
    parser.add_argument("--domain", type=str, help="The domain to be considered", required=False)
    parser.add_argument("--model", type=str, help="insightface's model to use for face similarity search.", choices=["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"], default="buffalo_sc", required=False)
    parser.add_argument("--headless", action="store_true", help="Interactions are via command line only. --input is required.", required=False)
    parser.add_argument("--restore", action="store_true", help="Restore state from the state.txt file", required=False)
    parser.add_argument("--save_all", action="store_true", help="Save even the non-relevant pictures with faces on it and their similarity score, even if it's not good", required=False)
    
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
    if os.path.exists(os.path.join(args.output, "output")):
        i = 1
        while os.path.exists(output_dir):
            output_dir = os.path.join(args.output, f"output_{i}")
            i += 1
    else:
        output_dir = os.path.join(args.output, "output")
        

    # Create output directory
    os.makedirs(output_dir)

    return args.headless, args.input, output_dir, args.domain, args.restore, args.save_all, args.model

if __name__ == "__main__":
    
    # setup
    headless, input_file, output_dir, domain, restore, save_all, model_name = argument_handling()
    print(f"Headless mode: \t\t{headless}")
    print(f"Input file: \t\t{input_file}")
    print(f"Output directory: \t{output_dir}")
    print(f"Domain: \t\t{domain}")
    
    promising_dir = os.path.join(output_dir, "promising")
    os.mkdir(promising_dir)
    
    recognizer = Recognizer(model_name)
    scraper = WebScraper(domain)
    if restore:
        scraper.load_state()
    
    # get embedding of face to be searched for
    face_embedding = recognizer.get_faces(cv2.imread(input_file))[0].embedding
    
    consecutive_errors = 0
    iter = 0
    while True:
        # save state every 10 iterations
        iter += 1
        if iter % 10 == 0:
            scraper.save_state()
            print("State saved.")
        try:
            # if too many image urls are in the queue, don't search for more urls and image-urls and concentrate on sifting only through the found image-urls instead
            if len(scraper.image_queue) < 1000:
                new_imgs, new_links = scraper.search(20)
                print(f"new images: {new_imgs}/{len(scraper.image_queue)}, new links:{new_links}/{len(scraper.urls_to_visit)}")
            for _ in tqdm(range(len(scraper.image_queue))):
                img_url, img_found_url = scraper.get_next_image_info()
                                
                # download image from its url and get all faces in it
                img = get_image_from_url(img_url)
                faces_strangers = recognizer.get_faces(img)
                
                # if no faces were found in image -> skip
                if not faces_strangers:
                    continue
                
                # go through all faces and calculate the cosine similarity of each face with the face to be searched for
                # similarity will be ~0 to 1, 1 being the most similar.
                max_similarity = 0
                similarities = []
                for stranger_face in faces_strangers:
                    similarity = recognizer.cosine_similarity(face_embedding, stranger_face.embedding)
                    similarities.append(similarity)
                    max_similarity = max(max_similarity, similarity)
                
                
                # potentially save image. Either when it's promising or when the save_all option is turned on. make bounding boxes around faces in image
                if save_all:
                    for j in range(len(faces_strangers)):
                        draw_bounding_box(img, faces_strangers[j], similarities[j])
                    cv2.imwrite(os.path.join(output_dir, img_url.replace("/",".")), img)
                if max_similarity > 0.3:
                    for j in range(len(faces_strangers)):
                        draw_bounding_box(img, faces_strangers[j], similarities[j])
                    print(f"Found a promising face with similarity {max_similarity:.2f} in {img_found_url}. Saving image.")
                    cv2.imwrite(os.path.join(promising_dir, img_url.replace("/",".")), img)
                    with open(os.path.join(promising_dir, 'promising_faces.txt'), 'a') as f:
                        f.write(f"{max_similarity:.2f} {img_url} {img_found_url} \n")
            
            consecutive_errors = 0
                    
        except Exception as e:
            # exponential backoff if errors occur
            print(f"An error occurred: {e}")
            time.sleep(1.2**consecutive_errors)
            consecutive_errors += 1



