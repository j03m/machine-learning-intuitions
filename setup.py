from setuptools import setup

setup(
    name='machine_learning_intuition',
    version="0.2",
    author="Joe Mordetsky",
    author_email="jmordetsky@gmail.com",
    description="All sorts of ml doodads",
    packages=['machine_learning_intuition'],
    package_dir={"": "src"},
    tests_require=["pytest"],
    install_requires=[line.rstrip('\n') for line in open('requirements.txt')]
)