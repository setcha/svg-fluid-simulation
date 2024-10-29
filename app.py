from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
from PIL import Image
import numpy as np

from simulate import run_simulation
from utils import process_svg

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

    #try:
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
    # except Exception as e:
    #     print(f"Simulation error: {e}")  # Print exception details
    #     return jsonify({'success': False, 'message': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)
