from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ext_modules = [
    Extension(
        "monotonic_align", ["monotonic_align.pyx"], include_dirs=[numpy.get_include()]
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(ext_modules),
)
