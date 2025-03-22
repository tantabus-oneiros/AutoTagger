import os
import csv
import requests
from PIL import Image
from io import BytesIO
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

    # Process file paths
    for file_path in file_paths:
        processed_count += 1
        if progress_callback:
            continue_processing = progress_callback(processed_count, total_items)
            if not continue_processing:
                # Processing was cancelled
                break

        try:
            image = Image.open(file_path)
            tags, scores = tagger.process_image(image, transform, threshold)

            results_dict[file_path] = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'tags': tags,
                'scores': scores,
                'image': image.copy()  # Store a copy of the image for thumbnail display
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
    progress_callback: Optional[Callable[[int, int], bool]] = None
) -> List[Dict]:
    """Process images from a list of URLs and return results"""
    results = []
    total_urls = len(urls)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            tags, scores = tagger.process_image(image, transform, threshold)

            results.append({
                'url': url,
                'tags': tags,
                'scores': scores,
                'image': image.copy()  # Store a copy of the image for thumbnail display
            })
        except Exception as e:
            results.append({
                'url': url,
                'error': str(e)
            })
        if progress_callback:
            continue_processing = progress_callback(i + 1, total_urls)
            if not continue_processing:
                # Processing was cancelled
                break
    return results

def format_results_as_csv(results: List[Dict]) -> str:
    """Format the results as a CSV string"""
    output = []
    headers = ['source', 'tags', 'scores']
    output.append(','.join(headers))

    for result in results:
        if 'error' in result:
            source = result.get('filename', result.get('url', result.get('input', 'unknown')))
            output.append(f"{source},ERROR,{result['error']}")
        else:
            source = result.get('filename', result.get('url', result.get('path', 'unknown')))
            tags_str = '|'.join(result.get('tags', []))
            scores_str = '|'.join(map(str, result.get('scores', [])))
            output.append(f"{source},{tags_str},{scores_str}")

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
                    txtfile.write(f"Tags: {', '.join(result['tags'])}\n")
                    txtfile.write(f"Scores: {', '.join(map(str, result['scores']))}\n")

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
                    txtfile.write(f"Tags: {', '.join(result['tags'])}\n")
                    txtfile.write(f"Scores: {', '.join(map(str, result['scores']))}\n")

                img_filename = os.path.splitext(os.path.basename(result.get('filename', result.get('url', 'unknown'))))[0] + '.png'
                img_file_path = os.path.join(temp_dir, img_filename)
                result['image'].save(img_file_path, 'PNG')

        with zipfile.ZipFile(output_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)