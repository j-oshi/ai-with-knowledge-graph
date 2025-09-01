import os
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urljoin
import requests

def get_document_filenames(allowed_extensions=None):
    """
    Retrieves a list of all filenames in the folder, optionally filtered by extension.

    Args:
        allowed_extensions (list, optional): A list of file extensions to include
                                            in the result (e.g., ['.txt', '.py']).
                                            The comparison is case-insensitive.
                                            If None, all files are returned.

    Returns:
        list: A list of files name in folder. Returns an empty list if the folder
              does not exist or is empty.
    """
    folder_path = 'documents'
    filenames = []

    try:
        entries = os.listdir(folder_path)
        
        for entry in entries:
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                if allowed_extensions:
                    _, extension = os.path.splitext(entry)
                    if extension.lower() in [ext.lower() for ext in allowed_extensions]:
                        filenames.append(entry)
                else:
                    filenames.append(entry)
                    
    except FileNotFoundError:
        print(f"Error: The folder '{folder_path}' was not found.")
        return []

    return filenames

def get_sitemap_urls(base_url: str, sitemap_filename: str = "sitemap.xml") -> List[str]:
    """Fetches and parses a sitemap XML file to extract URLs.

    Args:
        base_url: The base URL of the website
        sitemap_filename: The filename of the sitemap (default: sitemap.xml)

    Returns:
        List of URLs found in the sitemap. If sitemap is not found, returns a list
        containing only the base URL.

    Raises:
        ValueError: If there's an error fetching (except 404) or parsing the sitemap
    """
    try:
        sitemap_url = urljoin(base_url, sitemap_filename)

        # Fetch sitemap URL
        response = requests.get(sitemap_url, timeout=10)

        # # Return just the base URL if sitemap not found
        if response.status_code == 404:
            return [base_url.rstrip("/")]

        response.raise_for_status()

        # Parse XML content
        root = ET.fromstring(response.content)

        # Handle different XML namespaces that sitemaps might use
        namespaces = (
            {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else ""
        )

        # Extract URLs using namespace if present
        if namespaces:
            urls = [elem.text for elem in root.findall(".//ns:loc", namespaces)]
        else:
            urls = [elem.text for elem in root.findall(".//loc")]

        return urls

    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch sitemap: {str(e)}")
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse sitemap XML: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing sitemap: {str(e)}")
    
def get_word_length(words: str):
    word_list = words.split()
    word_list = [x for x in word_list if x != ' ']
    num_words = len(word_list)
    return num_words