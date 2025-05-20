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

import unittest
import sys
import types

# Provide minimal stubs for external modules used by utils
if 'pandas' not in sys.modules:
    sys.modules['pandas'] = types.SimpleNamespace(DataFrame=object)
if 'pytz' not in sys.modules:
    sys.modules['pytz'] = types.SimpleNamespace(timezone=lambda tz: tz)

from utils import sanitize_input

class TestUtils(unittest.TestCase):
    def test_sanitize_input_preserves_punctuation(self):
        text = "Hello, world: (test) 1+1=2"
        self.assertEqual(sanitize_input(text), text)

    def test_sanitize_input_removes_nonprintable(self):
        text = "Hello\x00World"
        self.assertEqual(sanitize_input(text), "HelloWorld")

if __name__ == "__main__":
    unittest.main()
