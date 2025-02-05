# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys


_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.insert(0, _ROOT_DIR)


# -- Project information -----------------------------------------------------

_VERSION = {}
with open(os.path.join(_ROOT_DIR, "focalcodec", "version.py")) as f:
    exec(f.read(), _VERSION)

project = "FocalCodec"
copyright = "2025, Luca Della Libera"
author = "Luca Della Libera"

# The major project version
version = _VERSION["VERSION"]

# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme,
# see the documentation.
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "collapse_navigation": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/style.css"]

autoapi_type = "python"

autoapi_dirs = [os.path.join(_ROOT_DIR, "focalcodec")]

# Flatten the API structure so submodules appear directly in the sidebar
autoapi_root = "."  # Places all generated API docs in the root (no nested structure)

# AutoAPI options to display all members
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
    "show-module-summary",
]

# Prevent AutoAPI from adding an extra "API" section
autoapi_add_toctree_entry = False
autoapi_keep_files = True
