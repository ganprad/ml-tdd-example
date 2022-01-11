""" Setup file for project."""

from setuptools import setup, find_packages


setup(
    name="skeleton",
    version="0.0.1",
    license="MIT",
    description="A project skeleton",
    author="Pradeep Ganesan",
    packages=find_packages(where='src')
)
