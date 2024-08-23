from setuptools import setup, find_packages

setup(
    name='aemulusnu_hmf',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'classy',
        'pytest'
    ],
    include_package_data=True,
    package_data={'': ['data/*.txt']},
    author='Delon Shen',
    author_email='delon@stanford.edu',
    description='Halo Mass Function Emulator for wνCDM cosmologies based on the Aemulus-ν suite of N-body simulations',
)
