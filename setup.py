from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='travel-home',
      version="0.0.1",
      description="Travel home, find your travel destination at a train or car distance",
    #   license="MIT",
      author="Batch #1117",
      author_email="auvray.n@gmail.com",
      #url="https://github.com/nauvray/travel-home",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=False,
      zip_safe=False)
