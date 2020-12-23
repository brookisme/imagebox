import random
import numpy as np
from scipy.signal import convolve2d
from imagebox.config import FIRST, LAST, BAND_ORDERING, BANDS_FIRST
#
# CONSTANTS
#
IMAGE='image'
DEFAULT_VMAP_VALUE=IMAGE
BANDS_FIRST_AXES=(1,2)
BANDS_LAST_AXES=(0,1)
DENORM_ERROR='imagebox.processor.denormalize: bands last not yet implemented'
SWAP_BANDS_ERROR='imagebox.processor._swap_bands_axes: im.ndim must be 3 or 4'
SMOOTHING_KERNEL=np.ones((3,3))


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


def denormalize(im,means,stdevs,bands=[0,1,2],bands_first=BANDS_FIRST,dtype=np.uint8):
    """ denormalize image array
    Args:
        im<np.array>: image array
        means<np.array|list>: means
        stdevs<np.array|list>: stdevs
        bands_first<bool>: true if array is bands first
    """ 
    if not bands_first:
        im=im[:,:,bands]
        im=im.transpose(2,0,1)
    else:
        im=im[bands]
    stdevs=np.array(stdevs)[bands]
    means=np.array(means)[bands]
    stdevs=stdevs.reshape((im.shape[0],1,1))
    means=means.reshape((im.shape[0],1,1))
    im=stdevs*im+means
    if not bands_first:
        im=im.transpose(1,2,0)
    return im.astype(dtype)


def map_values(im,value_map,default_value=DEFAULT_VMAP_VALUE):
    """ map values of image array
    Args:
        im<np.array>: image array
        value_map<dict>: 
            - keys: new values in mapped_im
            - values<list>: list of values to be mapped to corresponding key  
        default_value<int|float|np.nan|'image'>: 
            if default_value is 'image' use image values for unmapped values
            else use default value for unmapped values
    """
    value_map=value_map.copy()
    default_value=value_map.pop('.default',default_value)
    if default_value==IMAGE:
        mapped_im=np.array(im).copy()
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



def categorical_smoothing(im,nb_categories,kernel=SMOOTHING_KERNEL):
    """ smooth categorical inputs
    """
    im=to_categorical(im,nb_categories)
    for i in range(nb_categories):
        im[i]=convolve2d(im[i],kernel,mode='same')
    return im.argmax(axis=0)


def crop(im,cropping,bands_first=BANDS_FIRST):
    """ crop image """
    if bands_first:
        if im.ndim==4:
            return im[:,:,cropping:-cropping,cropping:-cropping]
        elif im.ndim==3:
            return im[:,cropping:-cropping,cropping:-cropping]    
        elif im.ndim==2:
            return im[cropping:-cropping,cropping:-cropping]
        elif im.ndim==1:
            return im[cropping:-cropping]
    else:
        if im.ndim>=2:
            return im[cropping:-cropping,cropping:-cropping]
        else:
            return im[cropping:-cropping]



def pad(im,padding=1,axes=None,value=0):
    """ pad image along axes
    * im<np.array>: image to pad
    * axes<None|int|tuple>: 
        - axis or axes to pad
        - if None assume bands last
    * padding<int|tuple>: number of pixes to pad by. tuple for have asymmetric padding
    * value<number>: constant value to pad with
    """
    if isinstance(axes,int):
        axes=(axes)
    elif axes is None:
        if im.ndim==4:
            axes=(2,3)
        elif im.ndim==3:
            axes=(1,2)
        elif im.ndim==2:
            axes=(0,1)
        elif im.ndim==1:
            axes=(0)
    def pad_tupl(axis,pad_tupl):
        if axis in axes:
            return pad_tupl
        else:
            return (0,0)
    if isinstance(padding,int):
        width=(padding,padding)
    else:
        width=padding
    return np.pad(
        im,
        pad_width=[pad_tupl(im,width) for im in range(im.ndim)],
        mode='constant',
        constant_values=value)


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


def augment(im,k=False,flip=False,bands_first=BANDS_FIRST,random=False):
    """ augment (rotate/flip) image

    Args:
        im<np.array>: image array
        k<int|False>: number of 90 degree rotations 
        flip<bool>: flip or don't flip
        random<bool>: if true get augmentation first
    """
    if random:
        k, flip=augmentation()
    if k is not False:
        im=rotate(im,k,bands_first=bands_first)
    if flip is not False:
        im=flip_image(im,bands_first=bands_first)
    return im


def rotate(im,k,bands_first=BANDS_FIRST):
    """ rotate image
    Args:
        im<np.array>: image array
        k<int|False>: number of 90 degree rotations

    * PYTORCH HACK:
    * - im=im+0
    * - negative stride issue 
    * - https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/7
    """
    im=np.rot90(im,k,axes=_axes(im.ndim,bands_first))
    im=im+0
    return im


def flip_image(im,bands_first=BANDS_FIRST,axis=None):
    """ flip image
    Args:
        im<np.array>: image array

    * PYTORCH HACK:
    * - im=im+0
    * - negative stride issue 
    * - https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/7
    """
    if axis is None:
        if (im.ndim==2) or (not bands_first):
            axis=0
        else:
            axis=1
    im=np.flip(im,axis)
    im=im+0
    return im


#
# HELPERS
#
def rgb_rescale(
        im,
        bands=None,
        rgb_max=255,
        im_max=2500,
        dtype=np.uint8,
        bands_first=BANDS_FIRST):
    if bands_first:
        if bands:
            im=im[bands]
        else:
            im=im[:3]
    else:
        if im.ndim==4:
            if bands:
                im=im[:,:,:,bands]
            else:
                im=im[:,:,:,:3]
        else:
            if bands:
                im=im[:,:,bands]
            else:
                im=im[:,:,:3]
    if im_max:
        im=im.astype(np.float)*rgb_max/im_max
    im=im.clip(0,rgb_max)
    if dtype:
        im=im.astype(dtype)
    return im


def is_bands_first(im):
    """ checks if image is bands last or bands first

    Note: Assumes the number of bands is less than or equal to the 
    width of the image.

    Returns: True/False
    """
    shape=im.shape
    return shape[-3]<=shape[-1]


def to_bands_last(im):
    """ convert image to bands last
    """
    shape=im.shape
    if (im.ndim>2) and is_bands_first(im):
        im=_swap_bands_axes(im)
    return im


def to_bands_first(im):
    """ convert image to bands first
    """    
    shape=im.shape
    if (im.ndim>2) and not is_bands_first(im):
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
        im=im.transpose(1,2,0)
    elif ndim==4:
        im=im.transpose(0,2,3,1)
    else:
        raise ValueError(SWAP_BANDS_ERROR)
    return im


def _to_vector(arr):
    return np.array(arr).reshape(-1,1,1)

