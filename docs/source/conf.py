# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""
EOM Sphinx configuration script.

"""

import sphinx_rtd_theme

import eomee


project = "EOMEE"


copyright = "2021, Ayers Lab"


author = "Ayers Lab"


release = eomee.__version__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]


exclude_patterns = []


html_theme = "sphinx_rtd_theme"


html_static_path = []


templates_path = []


mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.6/MathJax.js"


mathjax_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/HTML-CSS"],
}
