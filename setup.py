from setuptools import setup

setup(
   name='cmp_facade',
   version='0.1.0',
   author='Marius-Constantin Dinu',
   author_email='dinu.marius-constantin@hotmail.com',
   packages=[
       "timm",
       "gensim",
       "openmim"
    ],
   scripts=[],
   url='https://github.com/Xpitfire/SSIW',
   license='LICENSE.txt',
   description='Demo App for semantic segementation',
   long_description=open('README.txt').read(),
   install_requires=[
   ],
)