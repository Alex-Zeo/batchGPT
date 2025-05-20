from file_processor import detect_file_type


def test_detect_file_type():
    assert detect_file_type('document.pdf') == 'pdf'
    assert detect_file_type('file.docx') == 'docx'
    assert detect_file_type('script.py') == 'py'
    assert detect_file_type('archive.zip') == 'zip'
    assert detect_file_type('index.html') == 'code'
    assert detect_file_type('notes.txt') == 'txt'
