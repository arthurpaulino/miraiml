from distutils.core import setup
from miraiml import __version__

VERSION = __version__
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name = "MiraiML",
    version = VERSION,
    packages = ["miraiml"],
    author = "Arthur Paulino",
    author_email = "arthurleonardo.ap@gmail.com",
    url = "https://github.com/arthurpaulino/miraiml",
    description = "An asynchronous engine for continuous & autonomous machine\
        learning, built for real-time usage",
    long_description = LONG_DESCRIPTION
)
