import setuptools

# with open('README.md', 'r') as fh:
#     long_description = fh.read()

long_description="""
  v0.2 Release Notes:
  \n> change base df object to remove NaNs from strindex categories
  \n> Make start_hurdle for dates; added labels
  \n> Added key3 subs and key3_sum
  \n> Added weighting to strindex values that occurred earlier via favor_earlier and factors_to_favor_earlier
  \n> fixed stringency and case issues with agg_to_country_level
  \n> moved agg_to_contry_level inside CaseStudy class and added country_level flag for use
  \n> added helpers module for namespacing issues
  \n> added scatter flow and bar charts chart methods
"""

requires = [
  'bokeh>=2.0.0',
  'matplotlib>=3.2.0',
  'numpy>=1.18.0',
  'pandas>=1.0.0',
  'requests>=2.23.0',
]

setuptools.setup(
    name='see19',
    version='0.2.0',
    author='Ryan Skene',
    author_email='rjskene83@gmail.com',
    description='An interface for visualizing and analysing the see19 dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
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