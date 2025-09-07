import os
import csv
import requests
from PIL import Image
from io import BytesIO, StringIO
import tempfile
import zipfile
import shutil
from typing import List, Dict, Tuple, Optional, Callable

def get_supported_extensions() -> List[str]:
    return ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def is_valid_url(url: str) -> bool:
    """Basic validation to check if URL has http/https prefix"""
    return url.startswith('http://') or url.startswith('https://')

def is_valid_path(path: str) -> bool:
    """Check if a string is a valid file path"""
    if not path:
        return False

    # Check if the file exists
    if not os.path.isfile(path):
        return False

    # Check if it has a supported extension
    ext = os.path.splitext(path)[1].lower()
    return ext in get_supported_extensions()

def process_urls_or_paths(
    inputs: List[str],
    tagger,
    transform,
    threshold: float,
    progress_callback: Optional[Callable[[int, int], bool]] = None
) -> List[Dict]:
    """Process images from a list of URLs or file paths and return results"""
    # Create a mapping to track original order
    input_map = {}
    valid_inputs = []
    results_dict = {}

    # Separate URLs and file paths while preserving order
    urls = []
    file_paths = []

    for idx, input_str in enumerate(inputs):
        input_str = input_str.strip()
        if not input_str:
            continue

        input_map[input_str] = idx

        if is_valid_url(input_str):
            urls.append(input_str)
            valid_inputs.append(input_str)
        elif is_valid_path(input_str):
            file_paths.append(input_str)
            valid_inputs.append(input_str)
        else:
            # Invalid input - add directly to results_dict
            results_dict[input_str] = {
                'input': input_str,
                'error': "Invalid URL or file path"
            }

    # Calculate total items for progress
    total_items = len(urls) + len(file_paths)
    processed_count = 0

    # Process URLs
    if urls:
        for url in urls:
            processed_count += 1
            if progress_callback:
                continue_processing = progress_callback(processed_count, total_items)
                if not continue_processing:
                    # Processing was cancelled
                    break

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                image = Image.open(BytesIO(response.content))
                tags, scores = tagger.process_image(image, transform, threshold)

                results_dict[url] = {
                    'url': url,
                    'tags': tags,
                    'scores': scores,
                    'image': image.copy()  # Store a copy of the image for thumbnail display
                }
            except Exception as e:
                results_dict[url] = {
                    'url': url,
                    'error': str(e)
                }

    # Process file paths in batches
    batch_size = 8
    continue_processing = True
    
    for i in range(0, len(file_paths), batch_size):
        if not continue_processing:
            break
            
        batch_paths = file_paths[i:i+batch_size]
        batch_images = []
        batch_errors = []
        
        # Load batch of images
        for file_path in batch_paths:
            processed_count += 1
            if progress_callback:
                continue_processing = progress_callback(processed_count, total_items)
                if not continue_processing:
                    break
            
            try:
                image = Image.open(file_path)
                batch_images.append((file_path, image))
            except Exception as e:
                batch_errors.append((file_path, str(e)))
        
        # Handle individual image loading errors
        for file_path, error in batch_errors:
            results_dict[file_path] = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'error': error
            }
        
        # Skip if no images or processing was cancelled
        if not batch_images or not continue_processing:
            continue
            
        # Prepare batch data
        paths = [item[0] for item in batch_images]
        images = [item[1] for item in batch_images]
        
        try:
            # Process the batch
            batch_results = tagger.process_images(images, transform, threshold)
            
            # Store successful results
            for idx, (tags, scores) in enumerate(batch_results):
                results_dict[paths[idx]] = {
                    'path': paths[idx],
                    'filename': os.path.basename(paths[idx]),
                    'tags': tags,
                    'scores': scores,
                    'image': images[idx].copy()
                }
        except Exception as e:
            # Handle batch processing errors
            error_msg = f"Batch processing error: {str(e)}"
            for file_path in paths:
                results_dict[file_path] = {
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'error': error_msg
                }
        except Exception as e:
            results_dict[file_path] = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'error': str(e)
            }

    # Reconstruct results in original order
    results = []
    for input_str in inputs:
        input_str = input_str.strip()
        if not input_str:
            continue

        if input_str in results_dict:
            results.append(results_dict[input_str])

    return results

def process_folder(
    folder_path: str,
    tagger,
    transform,
    threshold: float,
    progress_callback: Optional[Callable[[int, int], bool]] = None
) -> List[Dict]:
    """Process all images in a folder and return results"""
    results = []
    supported_extensions = get_supported_extensions()

    try:
        # Get list of files first
        image_files = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = os.path.splitext(filename)[1].lower()

            if os.path.isfile(file_path) and ext in supported_extensions:
                image_files.append((filename, file_path))

        # Process each file with progress updates
        total_files = len(image_files)
        for i, (filename, file_path) in enumerate(image_files):
            try:
                # Update progress and check for cancellation
                if progress_callback:
                    continue_processing = progress_callback(i + 1, total_files)
                    if not continue_processing:
                        # Processing was cancelled
                        break

                image = Image.open(file_path)
                tags, scores = tagger.process_image(image, transform, threshold)
                results.append({
                    'filename': filename,
                    'path': file_path,
                    'tags': tags,
                    'scores': scores,
                    'image': image.copy()  # Store a copy of the image for thumbnail display
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'path': file_path,
                    'error': str(e)
                })
    except Exception as e:
        return [{'error': f"Error processing folder: {str(e)}"}]

    return results

def process_urls(
    urls: List[str],
    tagger,
    transform,
    threshold: float,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    max_workers: int = 4,
    request_delay: float = 0.5
) -> List[Dict]:
    """Process images from a list of URLs with rate limiting and parallel processing"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    from queue import Queue

    results = []
    total_urls = len(urls)
    url_queue = Queue()
    last_request_time = 0

    # Worker function with rate limiting
    def process_single_url(url: str) -> Dict:
        nonlocal last_request_time
        current_time = time.time()
        elapsed = current_time - last_request_time
        
        # Enforce rate limiting
        if elapsed < request_delay:
            time.sleep(request_delay - elapsed)
        
        try:
            last_request_time = time.time()
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            tags, scores = tagger.process_image(image, transform, threshold)

            return {
                'url': url,
                'tags': tags,
                'scores': scores,
                'image': image.copy()
            }
        except Exception as e:
            return {
                'url': url,
                'error': str(e)
            }

    # Process URLs with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_url, url): url for url in urls}
        
        for i, future in enumerate(as_completed(futures)):
            if progress_callback:
                continue_processing = progress_callback(i + 1, total_urls)
                if not continue_processing:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
            
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    'url': futures[future],
                    'error': str(e)
                })

    return results

def format_results_as_csv(results: List[Dict]) -> str:
    """Format the results as a CSV string with semi-colon delimiter"""
    output = []
    # Add header
    output.append("image_url;tags")
    
    # Process each result
    for result in results:
        if 'error' in result:
            # Handle error case
            source = result.get('filename', result.get('url', result.get('input', 'unknown')))
            output.append(f"{source};ERROR: {result['error']}")
        else:
            # Handle success case
            source = result.get('filename', result.get('url', result.get('path', 'unknown')))
            tags_str = result.get('tags', '')
            output.append(f"{source};{tags_str}")
    
    return '\n'.join(output)

def save_csv_to_file(output_path: str, csv_content: str) -> None:
    """Save CSV content to a file"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvfile.write(csv_content)

def create_txt_files_zip(output_path: str, results: List[Dict]) -> None:
    """Create a zip file containing text files with tags and scores"""
    with tempfile.TemporaryDirectory() as temp_dir:
        for result in results:
            if 'error' not in result:
                filename = os.path.splitext(os.path.basename(result.get('filename', result.get('url', 'unknown'))))[0] + '.txt'
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as txtfile:
                    txtfile.write(result['tags'])

        with zipfile.ZipFile(output_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

def create_txt_and_images_zip(output_path: str, results: List[Dict]) -> None:
    """Create a zip file containing text files with tags and scores and images"""
    with tempfile.TemporaryDirectory() as temp_dir:
        for result in results:
            if 'error' not in result:
                txt_filename = os.path.splitext(os.path.basename(result.get('filename', result.get('url', 'unknown'))))[0] + '.txt'
                txt_file_path = os.path.join(temp_dir, txt_filename)
                with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
                    txtfile.write(result['tags'])

                img_filename = os.path.splitext(os.path.basename(result.get('filename', result.get('url', 'unknown'))))[0] + '.png'
                img_file_path = os.path.join(temp_dir, img_filename)
                result['image'].save(img_file_path, 'PNG')

        with zipfile.ZipFile(output_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

# Tag Translation Functions

def load_translations_from_csv(file_path: str) -> Dict[str, str]:
    """Load translations from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary mapping original tags to their translations
    """
    translations = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2 and row[1].strip():  # Only store non-empty translations
                translations[row[0]] = row[1]
    return translations

def save_translations_to_csv(file_path: str, translations: Dict[str, str]) -> None:
    """Save translations to a CSV file.
    
    Args:
        file_path: Path to save the CSV file
        translations: Dictionary mapping original tags to their translations
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["original", "translation"])
        for original, translation in translations.items():
            writer.writerow([original, translation])

def apply_translations(text: str, translations: Dict[str, str]) -> str:
    """Apply translations to a comma-separated string of tags.
    
    Args:
        text: Comma-separated string of tags
        translations: Dictionary mapping original tags to their translations
        
    Returns:
        Translated comma-separated string of tags
    """
    print(f"Applying translations to: '{text}'")
    print(f"Translations dictionary: {translations}")
    
    # Handle empty input
    if not text.strip():
        return ""
    
    # Split the input text into individual tags
    tags = [tag.strip() for tag in text.split(',') if tag.strip()]
    print(f"Split tags: {tags}")
    
    # Process each tag according to the translations
    translated_tags = []
    for tag in tags:
        print(f"Processing tag: '{tag}'")
        if tag in translations:
            translation = translations[tag]
            print(f"  Found translation: '{translation}'")
            if translation == '.':  # Special character for deletion
                print(f"  Deleting tag '{tag}'")
                # Skip this tag entirely
            else:
                # Add all translated tags (may be multiple comma-separated tags)
                translated_parts = [t.strip() for t in translation.split(',') if t.strip()]
                print(f"  Adding translated parts: {translated_parts}")
                translated_tags.extend(translated_parts)
        else:
            # Keep original tag if no translation exists
            print(f"  No translation found, keeping original: '{tag}'")
            translated_tags.append(tag)
    
    # Join the translated tags with commas
    result = ', '.join(translated_tags) if translated_tags else ""
    print(f"Final translated result: '{result}'")
    return result

def translate_txt_file(input_path: str, translations: Dict[str, str]) -> str:
    """Translate tags in a TXT file.
    
    Args:
        input_path: Path to the TXT file
        translations: Dictionary mapping original tags to their translations
        
    Returns:
        Translated content as a string
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    translated_content = apply_translations(content, translations)
    
    return translated_content

def translate_csv_file(input_path: str, translations: Dict[str, str]) -> str:
    """Translate tags in a CSV file (only the 'tags' column).
    
    Args:
        input_path: Path to the CSV file
        translations: Dictionary mapping original tags to their translations
        
    Returns:
        Translated content as a string
    """
    print(f"Translating CSV file: {input_path}")
    print(f"Number of translations: {len(translations)}")
    
    # Read the entire file as text
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Make a copy of the original content
    original_content = content
    
    # First, detect the delimiter
    delimiter = ','  # Default to comma
    for possible_delimiter in [',', ';', '\t', '|']:
        if possible_delimiter in content.split('\n')[0]:
            delimiter = possible_delimiter
            break
    
    print(f"Using delimiter: '{delimiter}'")
    
    # Split the content into lines
    lines = content.split('\n')
    if not lines:
        print("Empty file")
        return ""
    
    # Parse the header to find the tags column
    header_line = lines[0]
    # Handle quoted headers
    header_parts = []
    in_quotes = False
    current_part = ""
    for char in header_line:
        if char == '"':
            in_quotes = not in_quotes
            current_part += char
        elif char == delimiter and not in_quotes:
            header_parts.append(current_part)
            current_part = ""
        else:
            current_part += char
    header_parts.append(current_part)  # Add the last part
    
    # Find the tags column
    tags_col_idx = -1
    for i, col in enumerate(header_parts):
        # Remove quotes if present
        col_name = col.strip('"\'')
        if 'tags' in col_name.lower():
            tags_col_idx = i
            print(f"Found tags column at index {tags_col_idx}: '{col_name}'")
            break
    
    if tags_col_idx == -1:
        print("Warning: No 'tags' column found in CSV. Using default index 1.")
        tags_col_idx = 1  # Default to second column as fallback
    
    # Create a temporary file to write the translated content
    temp_file = tempfile.mktemp(suffix='.csv')
    print(f"Writing translated CSV to temporary file: {temp_file}")
    
    # Process the file line by line, preserving the original format
    with open(temp_file, 'w', encoding='utf-8') as f:
        # Write the header unchanged
        f.write(lines[0] + '\n')
        
        # Process each data line
        for i in range(1, len(lines)):
            line = lines[i]
            if not line.strip():
                f.write(line + '\n' if i < len(lines) - 1 else line)
                continue
            
            # Parse the line, preserving quotes and delimiters
            parts = []
            in_quotes = False
            current_part = ""
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                    current_part += char
                elif char == delimiter and not in_quotes:
                    parts.append(current_part)
                    current_part = ""
                else:
                    current_part += char
            parts.append(current_part)  # Add the last part
            
            # Translate the tags column if it exists
            if len(parts) > tags_col_idx:
                tags_part = parts[tags_col_idx]
                
                # Check if the tags are quoted
                if tags_part.startswith('"') and tags_part.endswith('"'):
                    # Remove the quotes for translation
                    tags = tags_part[1:-1]
                    translated_tags = apply_translations(tags, translations)
                    # Add the quotes back
                    parts[tags_col_idx] = f'"{translated_tags}"'
                else:
                    # No quotes
                    translated_tags = apply_translations(tags_part, translations)
                    parts[tags_col_idx] = translated_tags
                
                print(f"Row {i}: Translated tags")
            
            # Join the parts back together with the original delimiter
            translated_line = delimiter.join(parts)
            f.write(translated_line + '\n' if i < len(lines) - 1 else translated_line)
    
    # Read the translated file
    with open(temp_file, 'r', encoding='utf-8') as f:
        translated_content = f.read()
    
    print(f"Translated CSV file size: {len(translated_content)} bytes")
    
    # Verify that only the tags column was changed
    if len(original_content.split('\n')) != len(translated_content.split('\n')):
        print("Warning: Line count changed during translation")
    
    return translated_content

def translate_txt_folder(folder_path: str, translations: Dict[str, str]) -> List[Dict]:
    """Translate all TXT files in a folder.
    
    Args:
        folder_path: Path to the folder containing TXT files
        translations: Dictionary mapping original tags to their translations
        
    Returns:
        List of dictionaries with filename and translated content
    """
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            translated_content = translate_txt_file(file_path, translations)
            results.append({
                'filename': filename,
                'content': translated_content
            })
    return results

def create_translated_txt_files_zip(output_path: str, results: List[Dict]) -> None:
    """Create a zip file containing translated TXT files.
    
    Args:
        output_path: Path to save the ZIP file
        results: List of dictionaries with filename and translated content
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for result in results:
            file_path = os.path.join(temp_dir, result['filename'])
            with open(file_path, 'w', encoding='utf-8') as txtfile:
                txtfile.write(result['content'])
        
        with zipfile.ZipFile(output_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)