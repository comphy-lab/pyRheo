from setuptools import setup, find_packages

setup(
    name='pyRheo',
    version='1.0.3',
    packages=find_packages(),  # Automatically discovers all packages and sub-packages
    install_requires=[
        'scipy',
        'scikit-optimize',
        'numpy',
        'joblib',
        'scikit-learn',
        'matplotlib',
        'pandas',
    ],
    include_package_data=True,
    package_data={
        'pyRheo.mlp_models': ['*.joblib'],    # Include .joblib files in mlp_models
        'pyRheo.pca_models': ['*.joblib'],    # Include .joblib files in pca_models

    },
    author='Isaac Y. Miranda Valdez',
    author_email='isaac.mirandavaldez@aalto.fi',
    description='A package for rheological modeling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://github.com/mirandi1/pyRheo',
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU License',
        'Operating System :: OS Independent',
    ],
)

