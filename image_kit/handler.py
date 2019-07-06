import numpy as np
import image_kit.io as io
import image_kit.processor as proc
import image_kit.indices as indices
#
# CONSTANTS
# 
INPUT_DTYPE=np.float
TARGET_DTYPE=np.int64
TO_CATEGORICAL_ERROR='image_kit.handler: nb_categories required for to_categorical'


#
# InputTargetHandler
#
class InputTargetHandler(object):
    """ 
    
    A handler for processing multiple input/target pairs in the 
    same way. 

    This is especially useful for constructing data loader/generators

    Args:
        input_bands<list|None>: list of bands to select from image
        means,stdevs<list|np.array>:
            - means/stdevs for all bands in input image (before selecting input_bands)
            - if means and not stdevs: center input around means
            - if means and stdevs normalize inputs
        band_indices<list>:
            - list of arguments for image_kit.indices.index
            - adds band_index bands to image
        value_map<dict>: 
            - target value map. 
            - keys: new values
            - values: list of values to map to new value 
        default_mapped_value<'image'|int|float>:
            - only used if value_name is not None
            - if 'image' keep all unmapped values
            - otherwise value for all unmapped values
        to_categorical<bool>: one-hot target image 
        nb_categories<int>: 
            - number of target categories
            - required for to_categorical
        augment<bool>: augment image
        input_cropping<int|None>: amount to crop input image
        target_cropping<int|None>: amount to crop target image
        input_dtype<str>: input data type
        target_dtype<str>: target data type

    """ 
    def __init__(self,
            input_bands=None,
            means=None,
            stdevs=None,
            band_indices=None,
            value_map=None,
            default_mapped_value=proc.DEFAULT_VMAP_VALUE,
            to_categorical=False,
            nb_categories=None,
            augment=True,
            input_cropping=None,
            target_cropping=None,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE ):
        self.input_bands=input_bands
        self.band_indices=band_indices
        self.value_map=value_map
        self.default_mapped_value=default_mapped_value
        self.means=means
        self.stdevs=stdevs
        if to_categorical and (not nb_categories):
            raise ValueError(TO_CATEGORICAL_ERROR)
        self.to_categorical=to_categorical
        self.nb_categories=nb_categories
        self._set_augmentation(augment)
        self.input_cropping=input_cropping
        self.target_cropping=target_cropping
        self.input_dtype=input_dtype
        self.target_dtype=target_dtype


    def input(self,path,return_profile=False):
        self.input_path=path
        im,profile=io.read(path)
        im=process_input(
            im,
            input_bands=self.input_bands,
            band_indices=self.band_indices,
            cropping=self.input_cropping,
            means=self.means,
            stdevs=self.stdevs,
            dtype=self.input_dtype )
        return self._return_data(
            im,
            profile,
            return_profile )


    def target(self,path,return_profile=False):
        self.target_path=path
        im,profile=io.read(path)
        im=process_target(
            im,
            value_map=self.value_map,
            default_mapped_value=self.default_mapped_value,
            categorical=self.to_categorical,
            nb_categories=self.nb_categories,
            cropping=self.target_cropping,
            dtype=self.target_dtype )
        return self._return_data(
            im,
            profile,
            return_profile )


    #
    # INTERNAL METHODS
    #
    def _set_augmentation(self,augment):
        self.augment=augment
        if augment:
            self.k, self.flip=proc.augmentation()
        else:
            self.k=False
            self.flip=False



    def _return_data(self,im,profile,return_profile):
        if self.augment:
            im=proc.augment(im,self.k,self.flip)
        if return_profile:
            return im, profile
        else:
            return im


#
# HELPERS
#
def process_input(
        im,
        input_bands=None,
        band_indices=None,
        cropping=None,
        means=None,
        stdevs=None,
        dtype=INPUT_DTYPE):
    if cropping:
        im=proc.crop(im,cropping)
    if band_indices:
        index_bands=[indices.index(im,idx) for idx in band_indices]
    if means is not None:
        if stdevs is None:
            im=proc.center(im,means=means,to_int=False)
        else:
            im=proc.normalize(im,means=means,stdevs=stdevs)
    if input_bands:
        im=im[input_bands]
    if band_indices:
        if input_bands is False:
            im=np.vstack([index_bands])
        else:
            im=np.vstack([im,index_bands])
    return im.astype(dtype)


def process_target(
        im,
        value_map=None,
        default_mapped_value=proc.DEFAULT_VMAP_VALUE,
        categorical=False,
        nb_categories=None,
        cropping=None,
        dtype=TARGET_DTYPE):
    if im.ndim==3:
        if proc.is_bands_first(im):
            im=im[0]
        else:
            im[:,:,0]
    if value_map:
        im=proc.map_values(im,value_map)
    if categorical:
        im=proc.to_categorical(im,nb_categories)
    if cropping:
        im=proc.crop(im,cropping)
    return im.astype(dtype)



