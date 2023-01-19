"""
https://packaging.python.org/tutorials/packaging-projects/

"""
import os
import setuptools

from leader_follower import project_properties

if os.path.exists('README.md'):
    with open('README.md', 'r') as desc_file:
        long_description = desc_file.read()
else:
    long_description = ''

if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as req_file:
        requirements_list = req_file.readlines()
else:
    requirements_list = []

setuptools.setup(
    name=project_properties.name,
    version=project_properties.version,

    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',

    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=setuptools.find_packages(),
    install_requires=requirements_list
)
