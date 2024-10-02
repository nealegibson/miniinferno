from setuptools import setup

setup(
  
  name = "miniinferno",
  description='A tiny MCMC for python astro labs',  
  version = "1.0",
  author='Neale Gibson',
  author_email='n.gibson@tcd.ie',
  python_requires='>=3',
  
  packages=['miniinferno'],
  package_dir={'miniinferno':'src'},
  install_requires=['numpy','scipy','tqdm','dill'],
  )
