import random
import numpy as np
#
# CONSTANTS
#
IMAGE='image'
DEFAULT_VMAP_VALUE=IMAGE
BANDS_FIRST_AXES=(1,2)
BANDS_LAST_AXES=(0,1)
BANDS_FIRST=True
DENORM_ERROR='image_kit.processor.denormalize: bands last not yet implemented'
SWAP_BANDS_ERROR='image_kit.processor._swap_bands_axes: im.ndim must be 3 or 4'



#
# METHODS
#
def center(im,means=None,to_int=False,bands_first=BANDS_FIRST):
    """ center image array
    Args:
        im<np.array>: image array
        means<np.array|list>: 
            - band-wise means to center the array around
            - if none use the band-wise means of the image itself
        to_int<bool>: if true convert to uint8 after centering
        bands_first<bool>: true if array is bands first
    """
    if means is None:
        means=np.mean(im,axis=_axes(im.ndim,bands_first))
    if bands_first:
        means=_to_vector(means)
    im=(im-means)
    if to_int:
        im=im.round().astype(np.uint8)
    return im


def normalize(im,means=None,stdevs=None,bands_first=BANDS_FIRST):
    """ normalize image array
    Args:
        im<np.array>: image array
        means<np.array|list>: 
            - band-wise means to center the array around
            - if none use the band-wise means of the image itself
        stdevs<np.array|list>: 
            - band-wise standard deviations
            - if none use the band-wise standard deviations of the image itself
        bands_first<bool>: true if array is bands first
    """
    if stdevs is None:
        stdevs=np.std(im,axis=_axes(im.ndim,bands_first))   
    im=center(im,means=means,to_int=False,bands_first=bands_first)
    if bands_first:
        stdevs=_to_vector(stdevs)
    return im/stdevs


def denormalize(im,means,stdevs,bands_first=BANDS_FIRST):
    """ denormalize image array
    Args:
        im<np.array>: image array
        means<np.array|list>: means
        stdevs<np.array|list>: stdevs
        bands_first<bool>: true if array is bands first
    """ 
    if bands_first:
        im=im[:3]
        stdevs=np.array(stdevs[:3])
        means=np.array(means[:3])
        stdevs=np.array(stdevs).reshape((im.shape[0],1,1))
        means=np.array(means).reshape((im.shape[0],1,1))
        im=stdevs*im+means
        return im.astype(np.uint8)
    else:
        raise NotImplementedError(DENORM_ERROR)


def map_values(im,value_map,default_value=DEFAULT_VMAP_VALUE):
    """ map values of image array
    Args:
        im<np.array>: image array
        value_map<dict>: 
            - band-wise means to center the array around
            - if none use the band-wise means of the image itself
        default_value<int|float|np.nan|'image'>: 
            if default_value is 'image' use image values for unmapped values
            else use default value for unmapped values
    """
    if default_value==IMAGE:
        mapped_im=im.copy()
    else:
        mapped_im=np.full_like(im,default_value)
    for k,v in value_map.items():
        mapped_im[np.isin(im,v)]=k
    return mapped_im


def to_categorical(im,nb_categories):
    """ to categorical
    map single band int valued images to multi-band binary value image
    """
    return np.eye(nb_categories)[:,im]


def crop(im,cropping):
    """ crop image
    """
    if im.ndim==4:
        return im[:,:,cropping:-cropping,cropping:-cropping]
    elif im.ndim==3:
        return im[:,cropping:-cropping,cropping:-cropping]    
    elif im.ndim==2:
        return im[cropping:-cropping,cropping:-cropping]
    elif im.ndim==1:
        return im[cropping:-cropping]


def augmentation(k=None,flip=None):
    """ get k(rotation), flip values for image augmentation

    For both k and flip passing None will randomly select a value
    for k and flip. otherwise the k, flip value will be returned

    Args:
        k<int|None|False>: number of 90 degree rotations
        flip<bool|None>: true/false to flip or not flip
    """
    if k is not False: 
        if not k: k=random.randint(0,3)
    if flip is not False:
        if not flip: flip=random.choice([True,False])
    return k,flip


def augment(im,k=False,flip=False,bands_first=BANDS_FIRST):
    """ augment (rotate/flip) image

    Args:
        im<np.array>: image array
        k<int|False>: number of 90 degree rotations 
        flip<bool>: flip or don't flip

    * PYTORCH HACK:
    * - im=im+0
    * - negative stride issue 
    * - https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/7
    """
    if k is not False:
        im=np.rot90(im,k,axes=_axes(im.ndim,bands_first))
    if flip is not False:
        if (im.ndim==2) or (not bands_first):
            axis=1
        else:
            axis=2
        im=np.flip(im,axis)
    im=im+0
    return im


#
# HELPERS
#
def is_bands_first(im):
    """ checks if image is bands last or bands first

    Note: Assumes the number of bands is less than or equal to the 
    width of the image.

    Returns: True/False
    """
    shape=im.shape
    ndim=im.ndim    
    return shape[-3]<=shape[-1]


def to_bands_last(im):
    """ convert image to bands last
    """
    shape=im.shape
    ndim=im.ndim
    if is_bands_first(im):
        im=_swap_bands_axes(im)
    return im


def to_bands_first(im):
    """ convert image to bands first
    """    
    shape=im.shape
    ndim=im.ndim
    if not is_bands_first(im):
        im=_swap_bands_axes(im)
    return im



#
# INTERNAL
#
def _axes(ndim,bands_first):
    if (ndim==2) or (not bands_first):
        return BANDS_LAST_AXES
    else:
        return BANDS_FIRST_AXES



def _swap_bands_axes(im):
    ndim=im.ndim
    if ndim==3:
        im=im.swapaxes(0,1).swapaxes(1,2)
    elif ndim==4:
        im=im.swapaxes(1,2).swapaxes(2,3)
    else:
        raise ValueError(SWAP_BANDS_ERROR)
    return im


def _to_vector(arr):
    return np.array(arr).reshape(-1,1,1)

