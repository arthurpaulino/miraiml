"""
This module is meant for internal usage.
"""

from random import random, sample
from threading import Thread
from math import ceil
import pickle

def load(path):
    """
    A clean `pickle.load` wrapper for binary files.

    Parameters
    ----------
    path : string
        The path of the binary file to be loaded.

    Returns
    -------
    obj : object
    """
    return pickle.load(open(path, 'rb'))

def dump(obj, path):
    """
    A clean `pickle.dump` wrapper for binary files with a small difference: it
    loops if writing the file fails.

    Parameters
    ----------
    obj : object
        The object to be dumped to the binary file.

    path : string
        The path of the binary file to be written.
    """
    while True:
        try:
            pickle.dump(obj, open(path, 'wb'))
            return
        except:
            sleep(.1)

def par_dump(obj, path):
    """
    Optimizes the process of writing objects on disc by triggering a thread.

    Parameters
    ----------
    obj : object
        The object to be dumped to the binary file.

    path : string
        The path of the binary file to be written.
    """
    Thread(target=lambda: dump(obj, path)).start()

def sample_random_len(lst):
    """
    Returns a sample of random size from the list `lst`. The minimum length of
    the returned list is 1.

    Parameters
    ----------
    lst : list
        A list containing the elements to be sampled.

    Returns
    -------
    sampled_lst : list
        The randomly sampled elements from `lst`.
    """
    if len(lst) == 0:
        return []
    return sample(lst, max(1, ceil(random()*len(lst))))
