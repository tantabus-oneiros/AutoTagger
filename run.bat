@echo off
REM Activate the virtual environment
call venv\Scripts\activate

REM Run the inference script
python inference_gradio.py

REM Deactivate the virtual environment
deactivate