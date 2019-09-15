from distutils.core import setup
from miraiml import __version__

VERSION = __version__
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="MiraiML",
    version=VERSION,
    packages=["miraiml"],
    author="Arthur Leonardo de Alencar Paulino",
    author_email="arthurleonardo.ap@gmail.com",
    url="https://github.com/arthurpaulino/miraiml",
    description="An asynchronous engine for continuous & autonomous machine\
        learning, built for real-time usage",
    long_description=LONG_DESCRIPTION,
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
