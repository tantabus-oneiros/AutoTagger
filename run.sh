#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the inference script
python inference_gradio.py

# Deactivate the virtual environment
deactivate