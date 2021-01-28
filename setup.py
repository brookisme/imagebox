from distutils.core import setup
setup(
  name = 'imagebox',
  packages = ['imagebox'],
  version = '0.0.0.16',
  description = 'ImageBox: python utilities for working with multispectral imagery',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/imagebox',
  download_url = 'https://github.com/brookisme/imagebox/tarball/0.1',
  keywords = ['python','geotiff','rasterio','image','io'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  install_requires=[
    'numpy',
    'rasterio',
    'pyyaml',
    'affine',
    'rasterio',
    'gcs_helpers>=0.0.0.22'
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)