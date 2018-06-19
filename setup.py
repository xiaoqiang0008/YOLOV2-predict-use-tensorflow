# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

if os.name =='nt' :
    ext_modules=[
        Extension("lib.nms",
            sources=["lib/nms.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("lib.cy_yolo2_findboxes",
            sources=["lib/cy_yolo2_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("lib.cy_yolo_findboxes",
            sources=["lib/cy_yolo_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

elif os.name =='posix' :
    ext_modules=[
        Extension("lib.nms",
            sources=["lib/nms.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("lib.cy_yolo2_findboxes",
            sources=["lib/cy_yolo2_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("libcy_yolo_findboxes",
            sources=["lib/cy_yolo_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        )
    ]

else :
    ext_modules=[
        Extension("lib.nms",
            sources=["lib/nms.pyx"],
            libraries=["m"] # Unix-like specific
        ),        
        Extension("lib.cy_yolo2_findboxes",
            sources=["libcy_yolo2_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        ),
        Extension("lib.cy_yolo_findboxes",
            sources=["lib/cy_yolo_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        )
    ]

setup(
	 name='myyolo',
     description='yolov2',
     license='GPLv3',
#    packages = find_packages(),
#	scripts = ['flow'],
    ext_modules = cythonize(ext_modules)
)