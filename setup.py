#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='nemo',
    version='0.1.0',
    description="Neural network for Emotion detection, in short as NEmo, is a python library designed to implement Bayesian deep learning algorisms to emotion database for human emotion detection. It offers an easy construction of two types of deep neural network to incorporate the eletronic signal from sensor and predict human emotion, like valance and arousal.",
    long_description=readme + '\n\n',
    author="Yang Liu and Tianyi Zhang",
    author_email='y.liu@esciencecenter.nl',
    url='https://github.com/geek-yang/NEmo',
    packages=[
        'nemo',
    ],
    package_dir={'nemo':
                 'nemo'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='nemo',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
)
