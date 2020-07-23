import os
import yaml
#
# CONSTANTS
#
IMAGE_BOX_CONFIG_FILE=f'{os.getcwd()}/imagebox.config.yaml'
FIRST='first'
LAST='last'
NOISY=os.environ.get('IMAGE_BOX_NOISE',False)


#
# CONFIG
#
try: 
    with open(IMAGE_BOX_CONFIG_FILE,'rb') as file:
        _config=yaml.safe_load(file)
        if NOISY:
            print(f'IMAGE_BOX: config file loaded ({IMAGE_BOX_CONFIG_FILE})')
except Exception as e:
    _config={}
    if NOISY:
        print(f'IMAGE_BOX: config file ({IMAGE_BOX_CONFIG_FILE}) not found')


BAND_ORDERING=_config.get(
    'band_ordering',
    os.environ.get('IMAGE_BOX_BAND_ORDERING',FIRST))


#
# DERIVED
#
BANDS_FIRST=BAND_ORDERING==FIRST
