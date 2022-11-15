#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

import hgcal_npz

setup(
    name          = 'hgcal_npz',
    version       = hgcal_npz.VERSION,
    license       = 'BSD 3-Clause License',
    description   = 'Light-weight interface for HGCAL training data.',
    url           = 'https://github.com/tklijnsma/hgcal_npz.git',
    author        = 'Thomas Klijnsma',
    author_email  = 'tklijnsm@gmail.com',
    py_modules    = ['hgcal_npz'],
    zip_safe      = False,
    scripts       = ['bin/hgcal_npz_ls', 'bin/hgcal_nano_to_npz']
    )