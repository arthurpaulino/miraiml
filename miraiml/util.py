"""
:mod:`miraiml.util` provides utility functions that are used by higher level
modules.
"""

from threading import Thread
import random as rnd
import pickle
import math

def load(path):
    """
    A clean `pickle.load` wrapper for binary files.

    :param path: The path of the binary file to be loaded.
    :type path: string

    :rtype: object
    :returns: The loaded object.
    """
    return pickle.load(open(path, 'rb'))

def dump(obj, path):
    """
    A clean `pickle.dump` wrapper for binary files with a small difference: it
    loops if writing the file fails.

    :param obj: The object to be dumped to the binary file.
    :type obj: object

    :param path: The path of the binary file to be written.
    :type path: string
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

    :param obj: The object to be dumped to the binary file.
    :type obj: object

    :param path: The path of the binary file to be written.
    :type path: string
    """
    Thread(target=lambda: dump(obj, path)).start()

def sample_random_len(lst):
    """
    Returns a sample of random size from the list ``lst``. The minimum length of
    the returned list is 1.

    :type lst: list
    :param lst: A list containing the elements to be sampled.

    :rtype: sampled_lst: list
    :returns: The randomly sampled elements from ``lst``.
    """
    if len(lst) == 0:
        return []
    return rnd.sample(lst, max(1, math.ceil(rnd.random()*len(lst))))
