import os
import csv
import requests
from PIL import Image
from io import BytesIO
import tempfile
import zipfile
import shutil
from typing import List, Dict, Tuple, Optional, Callable

def get_supported_extensions():
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
    valid_urls = [url.strip() for url in urls if url.strip() and is_valid_url(url.strip())]
    total_urls = len(valid_urls)
    
    for i, url in enumerate(valid_urls):
        # Update progress and check for cancellation
        if progress_callback:
            continue_processing = progress_callback(i + 1, total_urls)
            if not continue_processing:
                # Processing was cancelled
                break
            
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
    
    return results

def format_results_as_csv(results: List[Dict]) -> str:
    """Format results as CSV string with semi-colon delimiter"""
    csv_output = []
    
    # Add header
    csv_output.append("image_url;tags")
    
    # Process each result
    for result in results:
        if 'error' in result:
            # Handle error case
            if 'filename' in result:
                # For folder processing or file paths
                csv_output.append(f"{result['filename']};ERROR: {result['error']}")
            elif 'url' in result:
                # For URL processing
                csv_output.append(f"{result['url']};ERROR: {result['error']}")
            elif 'input' in result:
                # For invalid inputs
                csv_output.append(f"{result['input']};ERROR: {result['error']}")
            else:
                # Generic error
                csv_output.append(f"unknown;ERROR: {result['error']}")
        else:
            # Handle success case
            if 'filename' in result:
                # For folder processing or file paths
                csv_output.append(f"{result['filename']};{result['tags']}")
            elif 'url' in result:
                # For URL processing
                csv_output.append(f"{result['url']};{result['tags']}")
            else:
                # Shouldn't happen, but just in case
                source = result.get('path', 'unknown')
                csv_output.append(f"{source};{result['tags']}")
    
    return "\n".join(csv_output)

def create_csv_file(csv_content: str) -> str:
    """Create a temporary CSV file for download and return its path"""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.csv')
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(csv_content)
        return path
    except Exception as e:
        # If there's an error, make sure to close the file descriptor
        os.close(fd)
        raise e

def save_csv_to_file(csv_content: str, output_path: str) -> Tuple[bool, str]:
    """Save CSV content to a file"""
    try:
        with open(output_path, 'w', newline='') as f:
            f.write(csv_content)
        return True, f"Successfully saved to {output_path}"
    except Exception as e:
        return False, f"Error saving file: {str(e)}"

def create_txt_files_zip(results: List[Dict]) -> str:
    """Create a zip file containing .txt files with tags for each image"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "tags.zip")
    
    try:
        # Track filenames to handle duplicates
        used_filenames = set()
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                if 'error' in result:
                    continue
                
                # Get base filename without extension
                if 'filename' in result:
                    base_name = os.path.splitext(result['filename'])[0]
                else:
                    # For URLs, use the last part of the URL
                    url_parts = result['url'].split('/')
                    base_name = url_parts[-1].split('?')[0]  # Remove query params
                    base_name = os.path.splitext(base_name)[0]
                
                # Handle duplicate filenames
                txt_filename = f"{base_name}.txt"
                counter = 1
                while txt_filename in used_filenames:
                    txt_filename = f"{base_name} ({counter}).txt"
                    counter += 1
                
                used_filenames.add(txt_filename)
                
                # Write tags to text file
                txt_content = result['tags']
                zipf.writestr(txt_filename, txt_content)
        
        return zip_path
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def create_txt_and_images_zip(results: List[Dict]) -> str:
    """Create a zip file containing both .txt files and images"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "tags_and_images.zip")
    
    try:
        # Track filenames to handle duplicates
        used_filenames = set()
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                if 'error' in result:
                    continue
                
                # Get base filename without extension
                if 'filename' in result:
                    base_name = os.path.splitext(result['filename'])[0]
                    ext = os.path.splitext(result['filename'])[1]
                else:
                    # For URLs, use the last part of the URL
                    url_parts = result['url'].split('/')
                    base_name = url_parts[-1].split('?')[0]  # Remove query params
                    base_name, ext = os.path.splitext(base_name)
                
                # Handle duplicate filenames for txt
                txt_filename = f"{base_name}.txt"
                counter = 1
                while txt_filename in used_filenames:
                    txt_filename = f"{base_name} ({counter}).txt"
                    counter += 1
                
                used_filenames.add(txt_filename)
                
                # Write tags to text file
                txt_content = result['tags']
                zipf.writestr(txt_filename, txt_content)
                
                # Add image to zip
                if 'image' in result:
                    # Handle duplicate filenames for image
                    img_filename = f"{base_name}{ext}"
                    counter = 1
                    while img_filename in used_filenames:
                        img_filename = f"{base_name} ({counter}){ext}"
                        counter += 1
                    
                    used_filenames.add(img_filename)
                    
                    # Save image to temp file and add to zip
                    img_temp = os.path.join(temp_dir, img_filename)
                    result['image'].save(img_temp)
                    zipf.write(img_temp, img_filename)
                    os.remove(img_temp)  # Clean up temp file
                elif 'path' in result:
                    # For folder processing, add original image
                    img_filename = result['filename']
                    counter = 1
                    while img_filename in used_filenames:
                        base, ext = os.path.splitext(result['filename'])
                        img_filename = f"{base} ({counter}){ext}"
                        counter += 1
                    
                    used_filenames.add(img_filename)
                    zipf.write(result['path'], img_filename)
        
        return zip_path
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e