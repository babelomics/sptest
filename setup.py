from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
  # mandatory
  name="sptest",
  description='Spanish Test with 1k Genome training.',
  long_description=readme(),
  # mandatory
  version="0.1",
  # mandatory
  author_email="carlos.loucera@juntadeandalucia.es",
  packages=['sptest'],
  include_package_data=True,
  install_requires=['numpy', 'pandas', 'click', 'dotenv', 'scikit-learn'],
  entry_points={
    'console_scripts': ['sptest = sptest.cli:start']
  }
)
