import os
import sys

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('../..'))

project = 'Latent Self'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
html_theme = 'alabaster'

exclude_patterns = ['_build']
autodoc_mock_imports = [
    'torch',
    'PyQt6',
    'mediapipe',
    'pydantic',
    'pydantic_settings',
    'onnxruntime',
    'jsonschema',
    'yaml',
    'cv2',
    'apscheduler',
    'werkzeug',
    'appdirs',
    'numpy',
]
