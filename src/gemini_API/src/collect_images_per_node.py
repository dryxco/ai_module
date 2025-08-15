#!/usr/bin/env python3
"""
readme
Collects the '0000.png' image from each node's image folder and saves it
to a new directory called 'image_per_node'.
"""

import os
import shutil

def collect_images_per_node():

    # Construct the absolute path to the ai_module directory
    # This assumes the script is located in a subdirectory of ai_module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the ai_module directory
    ai_module_dir = os.path.abspath(os.path.join(script_dir, "..", "..",".."))
    data_dir = os.path.join(ai_module_dir, "data")
    dest_dir = os.path.join(data_dir, "image_per_node")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    # Find all directories in the data directory that are named with integers
    try:
        node_dirs = sorted(
            [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()],
            key=int
        )
    except FileNotFoundError:
        print(f"Error: The data directory '{data_dir}' was not found.")
        return

    # Loop through each node directory
    for node_name in node_dirs:
        source_image_path = os.path.join(data_dir, node_name, "image", "0000.png")
        
        # Check if the source image exists
        if os.path.exists(source_image_path):
            # The destination file will be named after the node folder (e.g., '0.png', '1.png')
            dest_image_path = os.path.join(dest_dir, f"{node_name}.png")
            
            print(f"Copying '{source_image_path}' to '{dest_image_path}'")
            shutil.copy(source_image_path, dest_image_path)
        else:
            print(f"Warning: '0000.png' not found in node '{node_name}'.")

    print("\nImage collection process complete.")
    print(f"All collected images are saved in: {dest_dir}")

if __name__ == "__main__":
    collect_images_per_node()
