# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import pysprite

setup(

    name='pysprite',

    version=pysprite.__version__,

    packages=find_packages(),

    author="Quentin André",

    author_email="quentin.andre@insead.edu",

    description="A Python 3 re-implementation of Heathers, Anaya, van der Zee, and Brown's 'Sample Parameter Reconstruction via Iterative TEchniques (SPRITE)'",

    long_description=open('README.md', encoding="utf-8").read(),

    install_requires=["numpy", "pandas", "matplotlib", "seaborn"],

    extras_require={
        "dev": ['pytest']
    },

    keywords=['statistics', 'error-detection', 'granularity', 'testing'],

    url='https://github.com/QuentinAndre/pysprite/',

    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6"
    ],

    license="MIT"
)
