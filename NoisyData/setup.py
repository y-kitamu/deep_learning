import os
from distutils.core import setup
from setuptools import find_packages

VERSION = "0.0.0"


def load_requirements(f):
    if not os.path.exists(f):
        print("No such file or directory: {}".format(f))
        return []
    return list(
        filter(None,
               [l.split("#", 1)[0].strip() for l in open(os.path.join(os.getcwd(), f)).readlines()]))


setup(name="noisydata",
      version=VERSION,
      install_requires=load_requirements(os.path.join(os.path.dirname(__file__), "../requirements.txt")),
      packages=["noisydata"])
