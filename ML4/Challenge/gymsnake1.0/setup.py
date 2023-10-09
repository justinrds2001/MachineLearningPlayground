from setuptools import setup, find_packages

setup(name='gymsnake',
      version='1.0',
      python_requires='>=3.10',
      description='A Gymnasium environment for multiplayer snake',
      classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Machine Learning :: Reinforcement Learning',
      ],
      url='https://bitbucket.org/ercoargante',
      author='Erco Argante',
      author_email='erco.argante@gmail.com',
      license='MIT',
      packages=find_packages("src"),
      package_dir={"": "src"},
      install_requires=[
          'numpy', 
          'matplotlib', 
          'gymnasium',
      ],
      zip_safe=False)
