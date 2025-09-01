import logging
from urllib.parse import urlparse
from docling.document_converter import DocumentConverter
from utils.document_utils import get_sitemap_urls

_log = logging.getLogger(__name__)

def get_site_content(url: str):
    """
    Retrieves content from a given URL and converts it to markdown and JSON.

    Args:
        url (str): The URL of the web page to retrieve content from.

    Returns:
        str: The markdown content of the page, or an error message if the
             operation fails.
    """
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        _log.error(f"Invalid URL provided: {url}")
        return "Error: The provided URL is not valid."

    try:
        converter = DocumentConverter()
        result = converter.convert(url)
        document = result.document
        markdown_output = document.export_to_markdown()
        json_output = document.export_to_dict()

        print(markdown_output)
        return markdown_output
    except Exception as e:
        _log.error(f"Failed to get and convert content for URL: {url}. Error: {e}")
        return ""
    
def scrap_site_content(url: str):
    """
    Retrieves content from a given URL and converts it to markdown and JSON.

    Args:
        url (str): The URL of the web page to retrieve content from.

    Returns:
        list: The markdown content of the page, or an error message if the
             operation fails.
    """
    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        _log.error(f"Invalid URL provided: {url}")
        return "Error: The provided URL is not valid."
    try:
        sitemap_urls = get_sitemap_urls(url)
        converter = DocumentConverter()
        conv_results_iter = converter.convert_all(sitemap_urls)

        docs = []
        for result in conv_results_iter:
            if result.document:
                document = result.document
                docs.append(document.export_to_markdown())
        return docs
    except Exception as e:
        _log.error(f"Failed to get and convert content for URL: {url}. Error: {e}")
        return ""