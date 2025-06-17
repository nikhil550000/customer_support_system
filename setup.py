from setuptools import setup, find_packages
setup(name='ecomerce_bot',
      version='0.0.1',
      author='nikhil',
      author_email='nikhilsai550000@gmail.com',
      packages=find_packages(),
      install_requires=[
          'requests',
          'beautifulsoup4',
            'langchain',
            'langchain_astradb'
      ])