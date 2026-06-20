import hashlib
from urllib.parse import urlparse

def url_normalisation(url):
    """
    Parse the url
    """
    parsed_url = urlparse(url)
    # parsed in form scheme, netloc, path, params, query, fragment
    scheme = parsed_url.scheme.lower() # scheme and netloc are not case sensitive
    netloc = parsed_url.netloc.lower()
    if parsed_url.path.endswith('/'): # standardise remove trailing /
        path = parsed_url.path[:-1]
    else:
        path = parsed_url.path
    # params,query, and fragment are discarded, as the url from the five sources are encoded based on their url path
    return f'{scheme}://{netloc}{path}'


def create_source_id(url):
    """
    Create a unique source ID for each url
    """
    normalised_url = url_normalisation(url)
    return hashlib.sha256(normalised_url.encode()).hexdigest()