import gradio as gr
import os
from typing import Callable, Dict, Any, Tuple

def create_interface(
    tagger,
    transform,
    process_folder_fn,
    process_urls_fn,
    format_csv_fn,
    save_csv_fn
):
    """Create the Gradio interface with tabs for single and batch processing"""
    
    # Single image processing functions
    def run_classifier(image, threshold):
        if image is None:
            return "", {}
        return tagger.process_image(image, transform, threshold)
    
    def clear_image():
        return tagger.clear()
    
    # Batch processing functions
    def process_folder_path(folder_path, threshold, progress=gr.Progress()):
        if not folder_path or not os.path.isdir(folder_path):
            return "Invalid folder path", ""
        
        def progress_callback(current, total):
            progress(current/total, f"Processing image {current}/{total}")
        
        results = process_folder_fn(folder_path, tagger, transform, threshold, progress_callback)
        csv_output = format_csv_fn(results)
        
        return csv_output, csv_output
    
    def process_url_list(url_text, threshold, progress=gr.Progress()):
        if not url_text:
            return "No URLs provided", ""
        
        urls = [url.strip() for url in url_text.split('\n') if url.strip()]
        if not urls:
            return "No valid URLs found", ""
        
        def progress_callback(current, total):
            progress(current/total, f"Processing URL {current}/{total}")
        
        results = process_urls_fn(urls, tagger, transform, threshold, progress_callback)
        csv_output = format_csv_fn(results)
        
        return csv_output, csv_output
    
    def save_csv(csv_content, save_path):
        if not csv_content:
            return "No content to save"
        
        if not save_path:
            return "Please provide a file path"
        
        success, message = save_csv_fn(csv_content, save_path)
        return message
    
    # Create the interface
    with gr.Blocks(css=".output-class { display: none; }") as demo:
        gr.Markdown("""
        ## Joint Tagger Project: JTP-PILOT² Demo **BETA**
        This tagger is designed for use on furry images (though may very well work on out-of-distribution images, potentially with funny results).  A threshold of 0.2 is recommended.  Lower thresholds often turn up more valid tags, but can also result in some amount of hallucinated tags.
        This tagger is the result of joint efforts between members of the RedRocket team, with distinctions given to Thessalo for creating the foundation for this project with his efforts, RedHotTensors for redesigning the process into a second-order method that models information expectation, and drhead for dataset prep, creation of training code and supervision of training runs.
        Special thanks to Minotoro at frosting.ai for providing the compute power for this project.
        """)
        
        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Source", sources=['upload'], type='pil', height=512, show_label=False)
                        threshold_slider = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.20, label="Threshold")
                    with gr.Column():
                        tag_string = gr.Textbox(label="Tag String")
                        label_box = gr.Label(label="Tag Predictions", num_top_classes=250, show_label=False)
                
                image_input.upload(
                    fn=run_classifier,
                    inputs=[image_input, threshold_slider],
                    outputs=[tag_string, label_box]
                )
                
                image_input.clear(
                    fn=clear_image,
                    inputs=[],
                    outputs=[tag_string, label_box]
                )
                
                threshold_slider.input(
                    fn=lambda t: tagger.create_tags(t),
                    inputs=[threshold_slider],
                    outputs=[tag_string, label_box]
                )
            
            # Batch Processing Tab
            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    batch_threshold = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.20, label="Threshold")
                
                with gr.Tabs():
                    # Folder Processing Tab
                    with gr.TabItem("Process Folder"):
                        folder_path = gr.Textbox(label="Folder Path", placeholder="Enter folder path containing images")
                        process_folder_btn = gr.Button("Process Folder")
                        folder_output = gr.Textbox(label="Results (CSV Format)", lines=10)
                        folder_csv = gr.Textbox(visible=False)
                        folder_save_path = gr.Textbox(label="Save Path", placeholder="Enter path to save CSV file")
                        folder_save_btn = gr.Button("Save to File")
                        folder_save_msg = gr.Textbox(label="Status")
                        
                        process_folder_btn.click(
                            fn=process_folder_path,
                            inputs=[folder_path, batch_threshold],
                            outputs=[folder_output, folder_csv]
                        )
                        
                        folder_save_btn.click(
                            fn=save_csv,
                            inputs=[folder_csv, folder_save_path],
                            outputs=[folder_save_msg]
                        )
                    
                    # URL Processing Tab
                    with gr.TabItem("Process URLs"):
                        url_input = gr.Textbox(label="Image URLs", placeholder="Enter one URL per line (must start with http:// or https://)", lines=5)
                        process_url_btn = gr.Button("Process URLs")
                        url_output = gr.Textbox(label="Results (CSV Format)", lines=10)
                        url_csv = gr.Textbox(visible=False)
                        url_save_path = gr.Textbox(label="Save Path", placeholder="Enter path to save CSV file")
                        url_save_btn = gr.Button("Save to File")
                        url_save_msg = gr.Textbox(label="Status")
                        
                        process_url_btn.click(
                            fn=process_url_list,
                            inputs=[url_input, batch_threshold],
                            outputs=[url_output, url_csv]
                        )
                        
                        url_save_btn.click(
                            fn=save_csv,
                            inputs=[url_csv, url_save_path],
                            outputs=[url_save_msg]
                        )
    
    return demo