#from setuptools import setup
import setuptools

def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(name='explibrosa',
      version='0.0.0.dev1',
      description='A tool to analyse and convert data coming from the Librosa python package.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/emrecdem/exploface',
      keywords='analysis, conversion, facial recoqnition, action units, Openface',
      author='V. van Hees and B.L. de Vries',
      author_email='b.devries@esciencecenter.nl',
      classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
      ),
      packages=setuptools.find_packages(),
      install_requires=['numpy', 'pandas', 'elanwriter', 'exploface'],
      zip_safe=True)