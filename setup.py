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
    install_requires=[
        'numpy==1.16.2', 'pandas==0.24.2', 'click==7.0', 'scikit-learn==0.20.3',
        'matplotlib==3.0.3', 'hyperopt==0.1.2', 'xgboost==0.90', 'joblib==0.13.2'
    ],
    entry_points={
        'console_scripts': ['sptest = sptest.cli:main']
    }
)
