import os
from utils import sanitize_input, calculate_batch_size, generate_hash


def test_sanitize_input():
    assert sanitize_input("Hello<script>") == "Helloscript"


def test_calculate_batch_size_small():
    assert calculate_batch_size(10, max_batch_size=100) == 10


def test_calculate_batch_size_large():
    assert calculate_batch_size(2500, max_batch_size=5000) == 2500


def test_generate_hash_consistency():
    assert generate_hash("abc") == generate_hash("abc")
    assert generate_hash("abc") != generate_hash("abcd")
