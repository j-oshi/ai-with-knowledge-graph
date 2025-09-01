from ingestion.extractor.html_extractor import get_site_content, scrap_site_content
from ingestion.vector import get_embedding_ollama, chunk_text


def site_to_vector(url: str, sitemap: bool = False):
    """
    Scrapes content from a URL, chunks it, and generates vector embeddings for each chunk.

    This function fetches content from a given URL. If a sitemap is specified, it will
    attempt to scrape content from all pages listed in the sitemap. Otherwise, it
    will scrape content from the single provided URL. The fetched content is then
    broken down into smaller, overlapping chunks. Finally, it uses an Ollama-hosted
    model to generate a vector embedding for each chunk.

    Args:
        url (str): The URL of the website or the sitemap to be processed.
        sitemap (bool, optional): If True, the function treats the URL as a sitemap
                                  and attempts to scrape all linked pages. If False,
                                  it only scrapes the single URL. Defaults to False.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a chunk
                    of the site's content with its corresponding vector embedding.
                    The format is:
                    [{'text': str, 'metadata_': {'doc': str, 'index': int}, 'embedding': list[float]}]
        list: An empty list if no content could be retrieved from the URL(s).
    """
    chunks = []
    if sitemap:
        site_content = scrap_site_content(url)
    else:
        site_content = get_site_content(url)

    if len(site_content) == 0:
        return []

    chunks = chunk_text(site_content, url)
    for chunki in chunks:
        embedding = get_embedding_ollama(chunki['text'])
        chunki['embedding'] = embedding
    return chunks