"""Sphinx configuration for gymnasium-retinatask documentation."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from datetime import datetime

# Project information
project = "gymnasium-retinatask"
copyright = f"{datetime.now().year}"
author = "Stefano Palmieri"

# Get version from package metadata
try:
    release = importlib.metadata.version(project)
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"

# Add src/ to path so autodoc finds the package
sys.path.insert(0, os.path.abspath("../src"))

# Sphinx extensions
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

# Intersphinx mapping to link to other documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# HTML theme
html_theme = "furo"
html_title = "Gymnasium Retina Task"
html_static_path = ["_static"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#7C4DFF",
        "color-brand-content": "#7C4DFF",
    },
    "dark_css_variables": {
        "color-brand-primary": "#B388FF",
        "color-brand-content": "#B388FF",
    },
}

# Template paths
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for HTML output
html_show_sourcelink = True
html_show_sphinx = False
