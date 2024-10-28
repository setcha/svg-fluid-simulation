import os
import imageio
import numpy as np
import subprocess
from PIL import Image

def make_gif_from_images(base_folder, identifier, gif_name, image_type='png', fps=50):

    filenames = [f for f in os.listdir(base_folder) if os.path.isfile(os.path.join(base_folder, f)) 
                 and f[-3:]==image_type
                 and f[0:len(identifier)]==identifier
                 ]
    file_numbers = [int(f.split(".")[0].split("_")[-1]) for f in filenames]
    f_name_numbers = sorted(zip(filenames, file_numbers), key=lambda x:x[1])
    filenames = [f[0] for f in f_name_numbers]

    images = []
    for filename in filenames:
        images.append(imageio.imread(os.path.join(base_folder, filename)))

    image_heights = [image.shape[0] for image in images]
    if min(image_heights) != max(image_heights):
        print("WARNING: resizing image heights to make a GIF")
        min_height = min(image_heights)
        #if the images arent consistent, then resize all images
        images = [image[-min_height:,:,:] for image in images]

    image_widths = [image.shape[1] for image in images]
    if min(image_widths) != max(image_widths):
        print("WARNING: resizing image widths to make a GIF")
        min_width = min(image_widths)
        #if the images arent consistent, then resize all images
        images = [image[:,0:min_width,:] for image in images]

    imageio.mimsave(gif_name, images, format='GIF', fps=fps)

def delete_files_in_directory(directory_path):
    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def process_svg(svg_path):
    """
    Processes the SVG file and creates separate geometry arrays for each color.

    Returns:
    - geometries: Dictionary mapping colors to NumPy arrays representing the geometry.
    """
    # Convert SVG to PNG using rsvg-convert
    png_filename = svg_path.replace('.svg', '.png')
    try:
        subprocess.run(["rsvg-convert", svg_path, "-o", png_filename], check=True)
    except subprocess.CalledProcessError:
        raise Exception("Error: 'rsvg-convert' failed. Please ensure it is installed and accessible.")

    # Load PNG image
    try:
        image = Image.open(png_filename).convert('RGBA')
        image_array = np.array(image)
    except Exception as e:
        raise Exception(f"Error loading PNG image: {e}")

    # Extract unique colors (excluding alpha = 0)
    pixels = image_array.reshape(-1, 4)
    # Only consider pixels where alpha > 0
    pixels = pixels[pixels[:, 3] > 0]
    unique_colors = np.unique(pixels[:, :3], axis=0)

    geometries = {}

    for color in unique_colors:
        # Create a mask for the current color
        mask = np.all(image_array[:, :, :3] == color, axis=-1)

        # Store the geometry array
        color_hex = '#{:02x}{:02x}{:02x}'.format(*color)
        geometries[color_hex] = mask.astype(np.uint8)

    return geometries