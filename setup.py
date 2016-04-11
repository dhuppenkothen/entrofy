from setuptools import setup, find_packages

import imp

version = imp.load_source('entrofy.version', 'entrofy/version.py')

setup(
    name='entrofy',
    version=version.version,
    description='Entrofy',
    author='Daniela Huppenkothen',
    author_email='daniela.huppenkothen@nyu.edu',
    url='http://github.com/dhuppenkothen/entrofy',
    download_url='http://github.com/dhuppenkothen/entrofy/releases',
    packages=find_packages(),
    long_description="""Entrofy""",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    license='ISC',
    install_requires=[
        'pandas>=0.16',
        'nump>=1.10'
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
