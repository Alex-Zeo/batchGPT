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
