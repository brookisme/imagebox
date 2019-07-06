from distutils.core import setup
setup(
  name = 'image_kit',
  packages = ['image_kit'],
  version = '0.0.0.1',
  description = 'Image Kit: Utilities for Reading/Writing images',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/image_kit',
  download_url = 'https://github.com/brookisme/image_kit/tarball/0.1',
  keywords = ['python','geotiff','rasterio','image','io'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)