"""
Shared non-fixture helpers for the nlp-search test suite.

Kept separate from conftest.py so test modules can import them directly
without triggering fixture auto-use side effects.
"""
from unittest.mock import MagicMock


def make_stream_chunks(*texts: str) -> list:
    """
    Build a list of MagicMock objects each with a .text attribute,
    simulating the Vertex AI streaming response iterated at
    [src/search/llm_service.py:77-78]:

        for response in responses:
            res_text_designation += response.text

    Usage:
        mock_generate_content.return_value = make_stream_chunks('["python"]')
        # or split across chunks to test concatenation:
        mock_generate_content.return_value = make_stream_chunks('["py', 'thon"]')
    """
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.text = text
        chunks.append(chunk)
    return chunks
