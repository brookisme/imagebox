import numpy as np
import image_kit.processor as proc
import image_kit.indices as idx


INPUT_DTYPE=np.float
TARGET_DTYPE=np.int64
TO_CATEGORICAL_ERROR='image_kit.handler: nb_categories required for to_categorical'


"""

TODO:

CHECK WORKING
CHECK INDICES
TODO: GRAB CORRECT DIMS

"""
#
# PROCESSING
#
def process_input(
        im,
        input_bands=None,
        indices=indices,
        cropping=None,
        dtype=INPUT_DTYPE):
    if indices:
        index_bands=[]
        for index in indices:
            index_bands.append(idx.index(im,index))
    if input_bands:
        im=im[input_bands]
    if indices:
        im=np.vstack([im,index_bands])
    if cropping:
        im=proc.crop(im,cropping)
    return im.astype(dtype)


def process_target(
        im,
        value_map=None,
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




#
# InputTargetHandler
#
class InputTargetHandler(object):
    #
    # PUBLIC METHODS
    #    
    def __init__(self,
            input_bands=None,
            normalize=True,
            indices=None,
            value_map=None,
            to_categorical=False,
            nb_categories=None
            augment=True,
            input_cropping=None,
            target_cropping=None,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE ):
        self.input_bands=input_bands
        self.indices=indices
        self.value_map=value_map
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
            indices=self.indices,
            cropping=self.input_cropping,
            dtype=self.input_dtype )
        return self._return_data(
            im,
            profile,
            return_profile )


    def target(self,path,return_profile=False):
        self.target_path=path
        cropping=self._cropping(crop)
        im,profile=io.read(path)
        im=process_target(
            im,
            value_map=self.value_map,
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
        if self.augment_data:
            im=proc.augment(im,self.k,self.flip)
        if return_profile:
            return im, profile
        else:
            return im
