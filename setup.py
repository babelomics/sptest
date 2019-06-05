from setuptools import setup, find_packages

setup(
      # mandatory
      name="sptest",
      # mandatory
      version="0.1",
      # mandatory
      author_email="carlos.loucera@juntadeandalucia.es",
      packages=['sptest'],
      package_data={},
      install_requires=['numpy', 'pandas', 'click', 'dotenv', 'scikit-learn'],
      entry_points={
        'console_scripts': ['sptest = sptest.cli:start']
      }
)
