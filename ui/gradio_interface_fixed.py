import gradio as gr
import utils.batch_processing
import os
import base64
import json
import csv
import tempfile
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
    print("Starting create_interface function")
    
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
        import os
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
        csv_file_path = os.path.join(tempfile.mkdtemp(), 'tags.csv')
        create_csv_file_fn(csv_file_path, csv_output)
        
        # Create TXT files zip
        txt_zip_path = os.path.join(tempfile.mkdtemp(), 'tags.zip')
        create_txt_files_zip_fn(txt_zip_path, results)
        
        # Create TXT and images zip
        all_zip_path = os.path.join(tempfile.mkdtemp(), 'tags_and_images.zip')
        create_txt_and_images_zip_fn(all_zip_path, results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    
    def process_url_list(url_list, threshold, progress=gr.Progress()):
        if not url_list:
            return "<p>No URLs provided</p>", None, None, None
        
        # Reset cancellation state at the start of processing
        global is_cancelled
        is_cancelled = False
        
        def progress_callback(current, total):
            progress(current/total, f"Processing URL {current}/{total}")
            # Check the cancel_processing state
            global is_cancelled
            return not is_cancelled
        
        results = process_urls_fn(url_list.split('\n'), tagger, transform, threshold, progress_callback)
        
        csv_output = format_csv_fn(results)
        html_output = format_results_as_html(results)
        
        # Create downloadable files
        csv_file_path = os.path.join(tempfile.mkdtemp(), 'tags.csv')
        create_csv_file_fn(csv_file_path, csv_output)
        
        # Create TXT files zip
        txt_zip_path = os.path.join(tempfile.mkdtemp(), 'tags.zip')
        create_txt_files_zip_fn(txt_zip_path, results)
        
        # Create TXT and images zip
        all_zip_path = os.path.join(tempfile.mkdtemp(), 'tags_and_images.zip')
        create_txt_and_images_zip_fn(all_zip_path, results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    
    def process_url_or_path_list(input_list, threshold, progress=gr.Progress()):
        if not input_list:
            return "<p>No URLs or paths provided</p>", None, None, None
        
        # Reset cancellation state at the start of processing
        global is_cancelled
        is_cancelled = False
        
        def progress_callback(current, total):
            progress(current/total, f"Processing item {current}/{total}")
            # Check the cancel_processing state
            global is_cancelled
            return not is_cancelled
        
        results = process_urls_or_paths_fn(input_list.split('\n'), tagger, transform, threshold, progress_callback)
        
        csv_output = format_csv_fn(results)
        html_output = format_results_as_html(results)
        
        # Create downloadable files
        csv_file_path = os.path.join(tempfile.mkdtemp(), 'tags.csv')
        create_csv_file_fn(csv_file_path, csv_output)
        
        # Create TXT files zip
        txt_zip_path = os.path.join(tempfile.mkdtemp(), 'tags.zip')
        create_txt_files_zip_fn(txt_zip_path, results)
        
        # Create TXT and images zip
        all_zip_path = os.path.join(tempfile.mkdtemp(), 'tags_and_images.zip')
        create_txt_and_images_zip_fn(all_zip_path, results)
        
        return html_output, csv_file_path, txt_zip_path, all_zip_path
    
    # Tag Translator Functions
    def dict_to_text(translations):
        """Convert translations dictionary to text format."""
        lines = []
        
        # Add header comment
        lines.append("# Tag Translations")
        lines.append("# Format: original_tag: translation")
        lines.append("# Use a period (.) to delete a tag")
        lines.append("# Leave empty to keep the original tag")
        lines.append("")
        
        # Add translations
        for original, translation in sorted(translations.items()):
            lines.append(f"{original}: {translation}")
        
        return "\n".join(lines)

    def text_to_dict(text):
        """Convert text format to translations dictionary."""
        translations = {}
        
        for line in text.split("\n"):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse line
            parts = line.split(":", 1)
            if len(parts) == 2:
                original = parts[0].strip()
                translation = parts[1].strip()
                
                if original:
                    if translation and translation != ".":
                        translations[original] = translation
                    elif translation == ".":
                        # Period means delete the tag
                        if original in translations:
                            del translations[original]
        
        return translations
    
    def load_all_tags():
        """Load all tags from tags.json with empty translations."""
        try:
            with open("tags.json", "r") as file:
                tags = json.load(file)
            
            # Create text with all tags
            lines = ["# Tag Translations", "# Format: original_tag: translation", "# Use a period (.) to delete a tag", "# Leave empty to keep the original tag", ""]
            
            # Add all tags with empty translations
            for tag in sorted(tags.keys()):
                # Replace underscores with spaces (as done in ImageTagger)
                display_tag = tag.replace("_", " ")
                lines.append(f"{display_tag}: ")
            
            return "\n".join(lines), f"Loaded all {len(tags)} tags from tags.json"
        except Exception as e:
            return "# Error loading tags", f"Error loading tags: {str(e)}"
    
    def load_translations_file(file_path):
        """Load translations from a CSV file."""
        try:
            translations = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) < 2:
                    return {}, "Invalid CSV format. Expected at least 2 columns."
                
                for row in reader:
                    if len(row) >= 2 and row[1].strip():  # Only store non-empty translations
                        translations[row[0]] = row[1]
            
            return translations, f"Loaded {len(translations)} translations successfully."
        except Exception as e:
            return {}, f"Error loading translations: {str(e)}"
            
    def load_translations_text(file_path):
        """Load translations from a CSV file and convert to text format."""
        translations, message = load_translations_file(file_path)
        text = dict_to_text(translations)
        return text, message
    
    def save_translations_text(text):
        """Convert text to translations dictionary and save to a CSV file."""
        translations = text_to_dict(text)
        
        if not translations:
            return None, "No translations to save. Please add some translations first."
        
        temp_file = tempfile.mktemp(suffix='.csv')
        status_message = save_translations_file(temp_file, translations)
        
        return temp_file, status_message
    
    def save_translations_file(file_path, translations):
        """Save translations to a CSV file."""
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["original", "translation"])
                for original, translation in translations.items():
                    writer.writerow([original, translation])
            
            return f"Saved {len(translations)} translations to {file_path}"
        except Exception as e:
            return f"Error saving translations: {str(e)}"
    
    def apply_translations_to_file(file_path, text):
        """Apply translations from text to a file."""
        translations = text_to_dict(text)
        
        if not translations:
            return None, "No translations to apply. Please add some translations first."
        
        return process_translate_file(file_path, translations)
    
    def process_translate_file(file_path, translations):
        """Translate tags in a file (TXT or CSV)."""
        try:
            if not file_path or not os.path.isfile(file_path):
                return None, "Invalid file path"
            
            if not translations:
                return None, "No translations provided"
            
            ext = os.path.splitext(file_path)[1].lower()
            
            # Create a temporary directory for the output file
            temp_dir = tempfile.mkdtemp()
            
            if ext == '.txt':
                translated_content = utils.batch_processing.translate_txt_file(file_path, translations)
                output_filename = os.path.basename(file_path).replace('.txt', '_translated.txt')
                output_path = os.path.join(temp_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                
                return output_path, f"Translated TXT file created"
            
            elif ext == '.csv':
                translated_content = utils.batch_processing.translate_csv_file(file_path, translations)
                output_filename = os.path.basename(file_path).replace('.csv', '_translated.csv')
                output_path = os.path.join(temp_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                
                return output_path, f"Translated CSV file created"
            
            else:
                return None, f"Unsupported file type: {ext}"
        
        except Exception as e:
            return None, f"Error translating file: {str(e)}"
    
    def process_translate_folder(folder_path, translations):
        """Translate all TXT files in a folder."""
        try:
            if not folder_path or not os.path.isdir(folder_path):
                return None, "Invalid folder path"
            
            if not translations:
                return None, "No translations provided"
            
            results = utils.batch_processing.translate_txt_folder(folder_path, translations)
            
            if not results:
                return None, "No TXT files found in the folder"
            
            # Create a zip file with translated TXT files
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, 'translated_tags.zip')
            utils.batch_processing.create_translated_txt_files_zip(zip_path, results)
            
            return zip_path, f"Translated {len(results)} TXT files"
        
        except Exception as e:
            return None, f"Error translating folder: {str(e)}"
    
    # Create the interface
    print("Creating Gradio Blocks")
    with gr.Blocks(css="""
        .output-class { display: none; }
        .results-container img { max-width: 250px; max-height: 250px; object-fit: contain; }
        .results-container a { text-decoration: none; }
        .results-container { margin-top: 10px; }
        
        /* Fix for table width issues */
        table.svelte-1adtv9j,
        table.svelte-1tckdwi,
        .table-wrap table,
        .gradio-container table {
            table-layout: fixed !important;
            width: 100% !important;
        }
        
        /* Make sure columns have appropriate widths */
        table.svelte-1adtv9j th:first-child,
        table.svelte-1adtv9j td:first-child,
        table.svelte-1tckdwi th:first-child,
        table.svelte-1tckdwi td:first-child,
        .table-wrap table th:first-child,
        .table-wrap table td:first-child,
        .gradio-container table th:first-child,
        .gradio-container table td:first-child {
            width: 50% !important;
        }
        
        table.svelte-1adtv9j th:last-child,
        table.svelte-1adtv9j td:last-child,
        table.svelte-1tckdwi th:last-child,
        table.svelte-1tckdwi td:last-child,
        .table-wrap table th:last-child,
        .table-wrap table td:last-child,
        .gradio-container table th:last-child,
        .gradio-container table td:last-child {
            width: 50% !important;
        }
        
        /* Ensure text wraps properly */
        table.svelte-1adtv9j td,
        table.svelte-1tckdwi td,
        .table-wrap table td,
        .gradio-container table td {
            word-break: break-word !important;
            white-space: normal !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        
        /* Fix for dataframe container */
        .gradio-container [data-testid="dataframe"] {
            overflow: hidden !important;
            max-width: 100% !important;
        }
        
        /* Improve status message visibility */
        .gradio-container [data-testid="markdown"] {
            font-weight: bold;
            color: #4CAF50;
        }
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
                
                with gr.Row():
                    classify_btn = gr.Button("Classify", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                # Set up event handlers for single image processing
                classify_btn.click(
                    fn=run_classifier,
                    inputs=[image_input, threshold_slider],
                    outputs=[tag_string, label_box]
                )
                
                clear_btn.click(
                    fn=clear_image,
                    inputs=[],
                    outputs=[image_input, tag_string, label_box]
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
            
            # Tag Translator Tab
            with gr.TabItem("Tag Translator"):
                gr.Markdown("""
                ## Tag Translator
                This tab allows you to create and apply tag translations. You can:
                - View and edit translations for tags
                - Save translations to a CSV file
                - Load translations from a CSV file
                - Apply translations to TXT or CSV files
                """)
                
                with gr.Tabs():
                    # Translation Management Tab
                    with gr.TabItem("Manage Translations"):
                        gr.Markdown("""
                        ### Tag Translation Management
                        
                        Edit translations in the text area below using the following format:
                        
                        ```
                        original_tag: translation
                        ```
                        
                        Special syntax:
                        - `original_tag: translation` - Translate a tag
                        - `original_tag: .` - Delete a tag
                        - `original_tag:` - Keep the original tag (no translation)
                        - `# Comment` - Add comments (will be ignored)
                        
                        Example:
                        ```
                        # My translations
                        anthro: anthropomorphic
                        female: female_character
                        male: male_character
                        # Delete this tag
                        unwanted_tag: .
                        ```
                        
                        Tips:
                        - Use your text editor's search function (Ctrl+F) to find specific tags
                        - Save your translations regularly
                        """)
                        
                        # State for storing translations
                        translation_state = gr.State({})
                        
                        # Text area for editing translations
                        translation_text = gr.Textbox(
                            label="Edit Translations",
                            placeholder="Enter translations in the format 'original_tag: translation' (one per line)",
                            lines=20,
                            max_lines=30,
                            interactive=True
                        )
                        
                        # Controls
                        with gr.Row():
                            load_all_tags_btn = gr.Button("Load All Tags")
                            load_btn = gr.Button("Load Translations")
                            save_btn = gr.Button("Save Translations")
                            translation_file = gr.File(label="Translation CSV File", file_types=[".csv"])
                        
                        # Status message
                        translation_status = gr.Markdown("")
                        
                        # Load all tags button handler
                        load_all_tags_btn.click(
                            fn=load_all_tags,
                            inputs=[],
                            outputs=[translation_text, translation_status]
                        )
                        
                        # Load translations from file
                        load_btn.click(
                            fn=lambda file: load_translations_text(file.name) if file else ("", "No file selected"),
                            inputs=[translation_file],
                            outputs=[translation_text, translation_status]
                        )
                        
                        # Save translations to file
                        save_btn.click(
                            fn=save_translations_text,
                            inputs=[translation_text],
                            outputs=[translation_file, translation_status]
                        )
                        
                        # Convert text to translations when needed
                        def update_translation_state(text):
                            """Update the translation state from text."""
                            translations = text_to_dict(text)
                            return translations
                        
                        # Update translation state when text changes
                        translation_text.change(
                            fn=update_translation_state,
                            inputs=[translation_text],
                            outputs=[translation_state]
                        )
                    
                    # Apply Translations Tab
                    with gr.TabItem("Apply Translations"):
                        gr.Markdown("""
                        ### Apply Translations to Files
                        - Upload a TXT or CSV file to apply translations
                        - For CSV files, translations are applied only to the tags column
                        - For TXT files, you can also process a folder containing multiple TXT files
                        
                        The translations from the "Manage Translations" tab will be used.
                        """)
                        
                        # Status message
                        apply_status = gr.Markdown("")
                        
                        # File input options
                        with gr.Tabs():
                            # Single file option
                            with gr.TabItem("Single File"):
                                input_file = gr.File(label="Input File (TXT/CSV)", file_types=[".txt", ".csv"])
                                translate_file_btn = gr.Button("Translate File")
                                translated_file = gr.File(label="Translated File", interactive=False)
                                
                                # Translate single file
                                translate_file_btn.click(
                                    fn=lambda file, text: apply_translations_to_file(file.name if file else None, text),
                                    inputs=[input_file, translation_text],
                                    outputs=[translated_file, apply_status]
                                )
                            
                            # Folder option (for TXT files)
                            with gr.TabItem("Folder (TXT files only)"):
                                folder_path_translate = gr.Textbox(label="Folder Path", placeholder="Enter folder path containing TXT files")
                                translate_folder_btn = gr.Button("Translate Folder")
                                translated_zip = gr.File(label="Translated Files (ZIP)", interactive=False)
                                
                                # Translate folder
                                translate_folder_btn.click(
                                    fn=lambda folder, text: process_translate_folder(folder, text_to_dict(text)),
                                    inputs=[folder_path_translate, translation_text],
                                    outputs=[translated_zip, apply_status]
                                )

    print("Returning demo:", demo)
    return demo