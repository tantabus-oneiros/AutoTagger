import gradio as gr
import os
import base64
from io import BytesIO
from typing import Callable, Dict, Any, Tuple, List

def format_results_as_html(results: List[Dict]) -> str:
    """Format results as HTML with image thumbnails and tags"""
    # Add a wrapper with a specific class for styling
    html_output = "<div class='results-container' style='display: flex; flex-direction: column; gap: 20px;'>"
    
    for result in results:
        if 'error' in result:
            # Handle error case
            error_source = result.get('filename', result.get('url', result.get('input', 'unknown')))
            html_output += f"<div><p>Error processing {error_source}: {result['error']}</p></div>"
        else:
            # Create a row with image thumbnail and tags
            html_output += "<div style='display: flex; gap: 15px; border-bottom: 1px solid #ddd; padding-bottom: 15px;'>"
            
            # Image thumbnail column (clickable)
            html_output += "<div style='flex: 0 0 250px;'>"
            # Use data URI for the image
            if 'image' in result:
                # Convert PIL image to data URI
                buffered = BytesIO()
                img = result['image'].copy()
                img.thumbnail((250, 250))  # Resize to thumbnail
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                data_uri = f"data:image/png;base64,{img_str}"
                
                # Create clickable image that opens full size in new tab
                source = result.get('path', result.get('url', ''))
                html_output += f"<a href='{source}' target='_blank'><img src='{data_uri}' style='max-width: 250px; max-height: 250px;' alt='{result.get('filename', result.get('url', ''))}'/></a>"
            html_output += "</div>"
            
            # Tags column
            html_output += "<div style='flex: 1;'>"
            html_output += f"<p><strong>{result.get('filename', result.get('url', ''))}</strong></p>"
            html_output += f"<p>{result['tags']}</p>"
            html_output += "</div>"
            
            html_output += "</div>"
    
    html_output += "</div>"
    return html_output

def create_interface(
    tagger,
    transform,
    process_folder_fn,
    process_urls_fn,
    process_urls_or_paths_fn,
    format_csv_fn,
    create_csv_file_fn,
    create_txt_files_zip_fn,
    create_txt_and_images_zip_fn
):
    """Create the Gradio interface with tabs for single and batch processing"""
    
    # Single image processing functions
    def run_classifier(image, threshold):
        if image is None:
            return "", {}
        return tagger.process_image(image, transform, threshold)
    
    def clear_image():
        return tagger.clear()
    
    # Use a simple variable for cancellation
    is_cancelled = False
    
    # Batch processing functions
    def process_folder_path(folder_path, threshold, progress=gr.Progress()):
        if not folder_path or not os.path.isdir(folder_path):
            return "<p>Invalid folder path</p>", None, None, None
        
        # Reset cancellation state at the start of processing
        global is_cancelled
        is_cancelled = False
        
        def progress_callback(current, total):
            progress(current/total, f"Processing image {current}/{total}")
            # Check the cancel_processing state
            global is_cancelled
            return not is_cancelled
        
        results = process_folder_fn(folder_path, tagger, transform, threshold, progress_callback)
        
        csv_output = format_csv_fn(results)
        html_output = format_results_as_html(results)
        
        # Create downloadable files
        csv_file_path = create_csv_file_fn(csv_output)
        txt_zip_path = create_txt_files_zip_fn(results)
        all_zip_path = create_txt_and_images_zip_fn(results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    def process_url_list(url_text, threshold, progress=gr.Progress()):
        if not url_text:
            return "<p>No URLs provided</p>", None, None, None
        
        urls = [url.strip() for url in url_text.split('\n') if url.strip()]
        if not urls:
            return "<p>No valid URLs found</p>", None, None, None
        
        # Reset cancellation state at the start of processing
        global is_cancelled
        is_cancelled = False
        
        def progress_callback(current, total):
            progress(current/total, f"Processing URL {current}/{total}")
            # Check the cancel_processing state
            global is_cancelled
            return not is_cancelled
        
        results = process_urls_fn(urls, tagger, transform, threshold, progress_callback)
        
        csv_output = format_csv_fn(results)
        html_output = format_results_as_html(results)
        
        # Create downloadable files
        csv_file_path = create_csv_file_fn(csv_output)
        txt_zip_path = create_txt_files_zip_fn(results)
        all_zip_path = create_txt_and_images_zip_fn(results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    
    def process_url_or_path_list(input_text, threshold, progress=gr.Progress()):
        if not input_text:
            return "<p>No inputs provided</p>", None, None, None
        
        inputs = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not inputs:
            return "<p>No valid inputs found</p>", None, None, None
        
        # Reset cancellation state at the start of processing
        global is_cancelled
        is_cancelled = False
        
        def progress_callback(current, total):
            progress(current/total, f"Processing item {current}/{total}")
            # Check the cancel_processing state
            global is_cancelled
            return not is_cancelled
        
        results = process_urls_or_paths_fn(inputs, tagger, transform, threshold, progress_callback)
        
        csv_output = format_csv_fn(results)
        html_output = format_results_as_html(results)
        
        # Create downloadable files
        csv_file_path = create_csv_file_fn(csv_output)
        txt_zip_path = create_txt_files_zip_fn(results)
        all_zip_path = create_txt_and_images_zip_fn(results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    
    # Create the interface
    with gr.Blocks(css="""
        .output-class { display: none; }
        .results-container img { max-width: 250px; max-height: 250px; object-fit: contain; }
        .results-container a { text-decoration: none; }
        .results-container { margin-top: 10px; }
    """) as demo:
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
                gr.Markdown("""
                ### ⚠️ Warning
                Avoid processing more than 100 images at once to prevent memory issues and long processing times.
                """)
                
                with gr.Row():
                    batch_threshold = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.20, label="Threshold")
                
                with gr.Tabs():
                    # Folder Processing Tab
                    with gr.TabItem("Process Folder"):
                        folder_path = gr.Textbox(label="Folder Path", placeholder="Enter folder path containing images")
                        
                        with gr.Row():
                            process_folder_btn = gr.Button("Process Folder", variant="primary")
                            cancel_folder_btn = gr.Button("Cancel", variant="stop")
                        
                        with gr.Row():
                            gr.Markdown("### Download Options")
                        
                        with gr.Row():
                            folder_csv = gr.File(label="CSV File", visible=True, interactive=False)
                            folder_txt_zip = gr.File(label="TXT Files (ZIP)", visible=True, interactive=False)
                            folder_all_zip = gr.File(label="TXT + Images (ZIP)", visible=True, interactive=False)
                        
                        folder_output = gr.HTML(label="Results")
                        
                        # Process folder button
                        process_folder_btn.click(
                            fn=process_folder_path,
                            inputs=[folder_path, batch_threshold],
                            outputs=[folder_output, folder_csv, folder_txt_zip, folder_all_zip]
                        )
                        
                        # Function to set the cancellation flag
                        def set_cancelled():
                            global is_cancelled
                            is_cancelled = True
                            return "Cancelling..."
                        
                        # Cancel button
                        cancel_folder_btn.click(
                            fn=set_cancelled,
                            inputs=[],
                            outputs=[gr.Textbox(visible=False)]
                        )
                    
                    # URL and Path Processing Tab
                    with gr.TabItem("Process URLs/Paths"):
                        url_input = gr.Textbox(
                            label="Image URLs or Paths",
                            placeholder="Enter one URL or file path per line\nURLs must start with http:// or https://\nPaths can be absolute (e.g., C:\\Images\\pic.png or /home/user/images/pic.jpg)",
                            lines=5
                        )
                        
                        with gr.Row():
                            process_url_btn = gr.Button("Process URLs/Paths", variant="primary")
                            cancel_url_btn = gr.Button("Cancel", variant="stop")
                        
                        with gr.Row():
                            gr.Markdown("### Download Options")
                        
                        with gr.Row():
                            url_csv = gr.File(label="CSV File", visible=True, interactive=False)
                            url_txt_zip = gr.File(label="TXT Files (ZIP)", visible=True, interactive=False)
                            url_all_zip = gr.File(label="TXT + Images (ZIP)", visible=True, interactive=False)
                        
                        url_output = gr.HTML(label="Results")
                        
                        # Process URLs/Paths button
                        process_url_btn.click(
                            fn=process_url_or_path_list,
                            inputs=[url_input, batch_threshold],
                            outputs=[url_output, url_csv, url_txt_zip, url_all_zip]
                        )
                        
                        # Cancel button
                        cancel_url_btn.click(
                            fn=set_cancelled,
                            inputs=[],
                            outputs=[gr.Textbox(visible=False)]
                        )
    
    return demo