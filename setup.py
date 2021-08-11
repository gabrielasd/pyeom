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
EOMEE setup script.

Run `python setup.py --help` for help.

"""

from eomee import __version__ as VERSION


NAME = "eomee"


# VERSION = "0.0.1"


LICENSE = "GPLv3"


AUTHOR = "Ayers Lab"


AUTHOR_EMAIL = "ayers@mcmaster.ca"


URL = "https://github.com/gabrielasd/eomee"


DESCRIPTION = "Equations of Motion and ERPA from molecular integrals and RDMs"


CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Topic :: Science/Engineering :: Molecular Science",
]


INSTALL_REQUIRES = [
    "numpy>=1.13",
    "scipy>=1.0",
]


EXTRAS_REQUIRE = {
    "test": ["pytest"],
    "docs": ["sphinx", "sphinx_rtd_theme"],
}


PACKAGES = [
    "eomee",
    "eomee.test",
]


PACKAGE_DATA = {
    "eomee.test": ["data/*.npy", "data/*.npz"],
}


INCLUDE_PACKAGE_DATA = True


ENTRY_POINTS = {
        "console_scripts": ["eom=eomee.scripts.run_eom:main",]
    }


if __name__ == "__main__":

    from setuptools import setup

    LONG_DESCRIPTION = open("README.rst", "r", encoding="utf-8").read()

    setup(
        name=NAME,
        version=VERSION,
        license=LICENSE,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        entry_points=ENTRY_POINTS,
    )
