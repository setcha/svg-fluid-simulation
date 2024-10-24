from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
from PIL import Image
import numpy as np

app = Flask(__name__)

# Ensure the outputs directory exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save_svg', methods=['POST'])
def save_svg():
    data = request.get_json()
    svg_data = data['svg']
    file_name = data['filename']
    boundaries = data['boundaries']
    color_velocities = data['colorVelocities']

    if not file_name.endswith('.svg'):
        file_name += '.svg'

    output_path = os.path.join('outputs', file_name)
    with open(output_path, 'w') as f:
        f.write(svg_data)

    # Save boundary conditions and color velocities
    json_output_path = os.path.join('outputs', file_name.replace('.svg', '.json'))
    with open(json_output_path, 'w') as f:
        json.dump({'boundaries': boundaries, 'colorVelocities': color_velocities}, f)

    print(f"Saved SVG and data as {file_name}")

    return f'SVG and boundary conditions saved successfully as {file_name}'

@app.route('/load_svg_data', methods=['POST'])
def load_svg_data():
    print("load_svg_data called")  # Print statement to verify route is called
    data = request.get_json()
    file_name = data['filename']

    svg_path = os.path.join('outputs', file_name)
    json_path = os.path.join('outputs', file_name.replace('.svg', '.json'))

    if not os.path.exists(svg_path) or not os.path.exists(json_path):
        return jsonify({'success': False, 'message': 'File not found.'})

    with open(svg_path, 'r') as f:
        svg_content = f.read()

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return jsonify({
        'success': True,
        'svgContent': svg_content,
        'boundaries': json_data.get('boundaries', {}),
        'colorVelocities': json_data.get('colorVelocities', {})
    })

@app.route('/simulate', methods=['POST'])
def simulate():
    print("simulate called")  # Print statement to verify route is called
    data = request.get_json()
    file_name = data['filename']

    svg_path = os.path.join('outputs', file_name)
    json_path = os.path.join('outputs', file_name.replace('.svg', '.json'))

    if not os.path.exists(svg_path) or not os.path.exists(json_path):
        return jsonify({'success': False, 'message': 'File not found.'})

    try:
        # Load SVG and JSON data
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Process SVG and create geometry arrays
        geometries = process_svg(svg_path)

        # Run the simulation
        results = run_simulation(geometries, json_data)

        # Optionally, save or return results
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Simulation error: {e}")  # Print exception details
        return jsonify({'success': False, 'message': str(e)})

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

def run_simulation(geometries, json_data):
    """
    Runs the simulation using the geometries and boundary conditions.

    Returns:
    - results: Simulation results (modify as needed).
    """
    # Access boundary conditions and color velocities
    boundaries = json_data['boundaries']
    color_velocities = json_data['colorVelocities']

    # Implement your simulation logic here
    # For example, you might iterate over the geometries and apply velocities

    # Placeholder for simulation results
    results = {}
    return results

if __name__ == '__main__':
    app.run(debug=True)
