from chunk_index import url_normalisation, create_source_id

def test_removing_trailing_slashes():
    url = "https://google.com/"
    expected = "https://google.com"

    assert url_normalisation(url) == expected

def test_removing_params_query_fragment():
    url = "https://google.com/path;param?query#fragment"
    expected = "https://google.com/path"

    assert url_normalisation(url) == expected

def test_case_sensitivity():
    url = "hTTPs://Google.com/path"
    expected = "https://google.com/path"   

    assert url_normalisation(url) == expected 