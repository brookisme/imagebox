import os
import yaml
#
# CONSTANTS
#
IMAGE_KIT_CONFIG_FILE=f'{os.getcwd()}/image_kit.config.yaml'
FIRST='first'
LAST='last'
NOISY=os.environ.get('IMAGE_KIT_NOISE',False)


#
# CONFIG
#
try: 
    with open(IMAGE_KIT_CONFIG_FILE,'rb') as file:
        _config=yaml.safe_load(file)
        if NOISY:
            print(f'IMAGE_KIT: config file loaded ({IMAGE_KIT_CONFIG_FILE})')
except Exception as e:
    _config={}
    if NOISY:
        print(f'IMAGE_KIT: config file ({IMAGE_KIT_CONFIG_FILE}) not found')


BAND_ORDERING=_config.get(
    'band_ordering',
    os.environ.get('IMAGE_KIT_BAND_ORDERING',FIRST))


#
# DERIVED
#
BANDS_FIRST=BAND_ORDERING==FIRST
