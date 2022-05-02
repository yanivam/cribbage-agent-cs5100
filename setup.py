from setuptools import setup, find_packages
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='cribbage',
    version='1.0.0',
    description='Cribbage Agent Game',
    long_description=long_description,
    url='https://github.com/yanivam/cribbage-agent-cs5100/',
    author='CS5100',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='cribbage cards card-games machine-learning ai',
    packages = ['cribbage'] ,
    package_data = {'cribbage': ['simple_model.pth', 'simple_pegging_model.pth']},
    include_package_data=True,
    install_requires=['numpy'],
    entry_points={
        'console_scripts': [
            'cribbage=cribbage.main:main',
        ],
    },
)
