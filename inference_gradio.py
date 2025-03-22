import torch
import json

# Import refactored modules
from models.image_tagger import ImageTagger
from utils.image_processing import create_transform
from utils.batch_processing import (
    process_folder,
    process_urls,
    process_urls_or_paths,
    format_results_as_csv,
    save_csv_to_file,
    create_csv_file,
    create_txt_files_zip,
    create_txt_and_images_zip
)
from ui.gradio_interface import create_interface

# Disable gradient computation for inference
torch.set_grad_enabled(False)

def main():
    # Initialize model and transform
    model_path = "JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors"
    tags_path = "tags.json"
    
    # Create tagger and transform
    tagger = ImageTagger(model_path, tags_path)
    transform = create_transform()
    
    # Create and launch the interface
    demo = create_interface(
        tagger=tagger,
        transform=transform,
        process_folder_fn=process_folder,
        process_urls_fn=process_urls,
        process_urls_or_paths_fn=process_urls_or_paths,
        format_csv_fn=format_results_as_csv,
        create_csv_file_fn=create_csv_file,
        create_txt_files_zip_fn=create_txt_files_zip,
        create_txt_and_images_zip_fn=create_txt_and_images_zip
    )
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main()