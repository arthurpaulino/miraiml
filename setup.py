from distutils.core import setup
from miraiml import __version__

VERSION = __version__
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

base_packages = ["scikit-learn>=0.21.1,<1", "pandas>=0.24.2,<1"]
dev_packages = ["sphinx>=2.0.1,<3", "sphinx-rtd-theme>=0.4.3,<1",
                "pytest>=5.1.2,<6", "flake8>=3.7.7,<4"]

setup(
    name="MiraiML",
    version=VERSION,
    packages=["miraiml"],
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
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
