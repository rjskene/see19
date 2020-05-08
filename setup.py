import setuptools

# with open('README.md', 'r') as fh:
#     long_description = fh.read()

requires = [
  'bokeh>=2.0.0',
  'matplotlib>=3.2.0',
  'numpy>=1.18.0',
  'pandas>=1.0.0',
  'requests>=2.23.0',
]

setuptools.setup(
    name='see19',
    version='0.1.8',
    author='Ryan Skene',
    author_email='rjskene83@gmail.com',
    description='An interface for the visualizing and analysing the see19 dataset',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    url='https://github.com/ryanskene/see19',
    packages=['see19'],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.7",
)