from distutils.core import setup
from miraiml import __version__

setup(
    name = "MiraiML",
    version = __version__,
    packages = ["miraiml"],
    author = "Arthur Paulino",
    author_email = "arthurleonardo.ap@gmail.com.com",
    description = "An asynchronous engine for autonomous & continuous machine\
        learning, built for real-time usage",
    url = "https://github.com/arthurpaulino/miraiml"
)
