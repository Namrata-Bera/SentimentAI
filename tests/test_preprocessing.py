from src.preprocessing import clean_text

def test_preserves_negation():
    assert 'not good' in clean_text('This is not good!')

def test_normalizes_url():
    assert 'URL' in clean_text('See https://example.com')

def test_handles_none():
    assert clean_text(None) == ''
