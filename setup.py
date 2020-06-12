import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

# long_description="""
# SUMMARY OF UPDATES

# \n1. Testset Graduation

# test counts and Apple mobility data have been moved into the main dataset.
# Reporting on testing continues to be inconsistent around the world. Many countries have only just begun reporting and many report on an infrequent basis (weekly or worse). Where there are gaps in daily figures, non-linear interpolation is used to smooth figures. Several key regions including Brazil and France have very minimal data at all.
# 2. Added filter functionality
# When instanting a CaseStudy instance:

# You can now pass any of region_id, region_code, or region_name to regions/exclude_regions in a single iterable. region_code column has been added, and is either simply a replica of country_code or the accepted abbreviation of the province or state. i.e. Alberta's region_code is AB.
# country_code and country_id now also acceptable in countries/exclude_countries
# pandas Series and numpy arrays are now acceptable iterables for these filters as well.
# 3. Miscellaneous

# To access the testset via get_baseframe, set test=True
# Added progress bar for get_baseframe() (a couple hours I won't ever get back)
# Additional styling attributes to most chart make() functions
# Added exception to catch when a country_w_sub is provided as region when country_level=False
# when USA is filter via countries, see19 now automatically excludes the country of Georgia. This was a major personal irritant of mine, but if you have the need you can simply include Georgia in countries as well.
# """

requires = [
  'bokeh>=2.0.0',
  'matplotlib>=3.2.0',
  'numpy>=1.18.0',
  'pandas>=1.0.0',
  'requests>=2.23.0',
]

setuptools.setup(
    name='see19',
    version='0.3.3',
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