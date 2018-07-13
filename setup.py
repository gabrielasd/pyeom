#!/usr/bin/env python


from io import open

from distutils.core import setup


if __name__ == '__main__':

    with open('README.md', mode='r', encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()

    setup(
        name='eomee',
        packages=['eomee'],
        requires=['numpy', 'scipy'],
        )
