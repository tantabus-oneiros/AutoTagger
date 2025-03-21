import os
import csv
import requests
from PIL import Image
from io import BytesIO
import tempfile
from typing import List, Dict, Tuple, Optional, Callable

def get_supported_extensions():
    return ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def is_valid_url(url: str) -> bool:
    """Basic validation to check if URL has http/https prefix"""
    return url.startswith('http://') or url.startswith('https://')

def process_folder(
    folder_path: str, 
    tagger, 
    transform, 
    threshold: float,
    progress_callback: Optional[Callable[[int, int], None]] = None
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
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_files)
                
                image = Image.open(file_path)
                tags, scores = tagger.process_image(image, transform, threshold)
                results.append({
                    'filename': filename,
                    'path': file_path,
                    'tags': tags,
                    'scores': scores
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
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict]:
    """Process images from a list of URLs and return results"""
    results = []
    valid_urls = [url.strip() for url in urls if url.strip() and is_valid_url(url.strip())]
    total_urls = len(valid_urls)
    
    for i, url in enumerate(valid_urls):
        # Update progress
        if progress_callback:
            progress_callback(i + 1, total_urls)
            
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            tags, scores = tagger.process_image(image, transform, threshold)
            
            results.append({
                'url': url,
                'tags': tags,
                'scores': scores
            })
        except Exception as e:
            results.append({
                'url': url,
                'error': str(e)
            })
    
    return results

def format_results_as_csv(results: List[Dict]) -> str:
    """Format results as CSV string"""
    csv_output = []
    
    # Check if results are from folder or URLs
    is_folder = 'filename' in results[0] if results else False
    
    if is_folder:
        csv_output.append("filename,tags")
        for result in results:
            if 'error' in result:
                csv_output.append(f"{result['filename']},ERROR: {result['error']}")
            else:
                csv_output.append(f"{result['filename']},{result['tags']}")
    else:
        csv_output.append("url,tags")
        for result in results:
            if 'error' in result:
                csv_output.append(f"{result['url']},ERROR: {result['error']}")
            else:
                csv_output.append(f"{result['url']},{result['tags']}")
    
    return "\n".join(csv_output)

def save_csv_to_file(csv_content: str, output_path: str) -> Tuple[bool, str]:
    """Save CSV content to a file"""
    try:
        with open(output_path, 'w', newline='') as f:
            f.write(csv_content)
        return True, f"Successfully saved to {output_path}"
    except Exception as e:
        return False, f"Error saving file: {str(e)}"