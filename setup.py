from setuptools import setup, find_packages

setup(
    name='aemulusnu_hmf',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
    author='Delon Shen',
    author_email='delon@stanford.edu',
    description='Halo Mass Function Emulator based on the Aemulus-nu suite of simulations',
)

