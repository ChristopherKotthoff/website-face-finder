import argparse
import tempfile
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk

import onnxruntime as onnx
#import onnxruntime-gpu as onnx


def argument_handling():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Search for similar faces on the web.")

    # Add arguments
    parser.add_argument("--headless", action="store_true", help="Interactions are via command line only. --input is required.", required=False)
    parser.add_argument("--input", type=str, help="Face image to look for", required=False)
    parser.add_argument("--output", type=str, help="City where the user lives", default=tempfile.gettempdir(), required=False)
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
    if len(args.domain.split(".")) < 2:
        print("Invalid domain. Should be like <example.com> or <subexample.example.com> or similar. Exiting...")
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

    return args.headless, args.input, output_dir, args.domain















if __name__ == "__main__":
    headless, input_file, output_dir, domain = argument_handling()

    print(f"Headless mode: \t\t{headless}")
    print(f"Input file: \t\t{input_file}")
    print(f"Output directory: \t{output_dir}")
    print(f"Domain: \t\t{domain}")


