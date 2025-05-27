from app import pdfreader


def test_pdfreader_wrapper_imports():
    assert hasattr(pdfreader, "load_pdf")
    assert hasattr(pdfreader, "chunk_pdf")
