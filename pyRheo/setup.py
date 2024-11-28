from setuptools import setup, find_packages

setup(
    name='pyRheo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'scikit-optimize',
        'numpy',
        'joblib',
        'scikit-learn',
        'matplotlib'  # add any other dependencies here
    ],
    include_package_data=True,
    package_data={
        # Include the joblib files in the package
        '': ['*.joblib'],
    },
    author='Isaac Y. Miranda Valdez',
    author_email='isaac.mirandavaldez@aalto.fi',
    description='A package for rheological modeling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/mirandi1/pyRheo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

