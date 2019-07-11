from random import randint
import numpy as np
import image_kit.io as io
import image_kit.processor as proc
import image_kit.indices as indices
#
# CONSTANTS
# 
INPUT_DTYPE=np.float
TARGET_DTYPE=np.int64
DEFAULT_SIZE=256
DEFAULT_OVERLAP=0
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
        cropping<int|None>:
            - amount to crop both input and target image when reading the image
        input_cropping<int|None>: 
            - amount to crop input image after processing the image
        target_cropping<int|None>: 
            - amount to crop target image after processing the image
        float_cropping<int|None>: 
            remove pixels from h/w of input and target starting at random i,j
        width<int|None>: image width (required if float cropping and not tiller)
        height<int|None>: image height (required if float cropping and not tiller)
        tiller<Tiller|True|None>:
            tile the image then read in a specific tile
        tiller_config<dict>:
            if tiller is True create tiller=Tiller(**tiller_config)
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
            cropping=None,
            input_cropping=None,
            target_cropping=None,
            float_cropping=None,
            width=None,
            height=None,
            tiller=None,
            tiller_config={},
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE ):
        if tiller is True:
            self.tiller=Tiller(**tiller_config)
        else:
            self.tiller=tiller
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
        self.augment=augment
        self.set_augmentation()
        self.cropping=cropping or 0
        self.input_cropping=input_cropping
        self.target_cropping=target_cropping
        self.float_cropping=float_cropping
        self.width=width
        self.height=height
        self.set_float_window()
        self.set_window()
        self.input_dtype=input_dtype
        self.target_dtype=target_dtype


    def input(self,path,return_profile=False):
        self.input_path=path
        im,profile=self._read(path)
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
        im,profile=self._read(path)
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


    def set_augmentation(self,k=None,flip=None):
        if self.augment:
            self.k, self.flip=proc.augmentation(k,flip)
        else:
            self.k=False
            self.flip=False
    

    def set_window(self,window_index=None):
        if self.tiller:
            if window_index is None:
                window_index=randint(0,len(self.tiller)-1)
            self.window_index=window_index
        else:
            self.window_index=False


    def set_float_window(self):
        if self.float_cropping:
            self.float_x=randint(0,2*self.float_cropping)
            self.float_y=randint(0,2*self.float_cropping)
        else:
            self.float_x=False
            self.float_y=False
    

    #
    # INTERNAL METHODS
    #
    def _read(self,path):
        if self.tiller:
            window=self.tiller[self.window_index]
        else:
            self._set_dimensions(path)
            window=None
        window=self._cropping_window(window)
        window=self._float_window(window)
        return io.read(path,window)


    def _cropping_window(self,window):
        if window:
            window=window[0]+self.cropping,window[1]+self.cropping,window[2]-2*self.cropping,window[3]-2*self.cropping
        elif self.cropping:
            window=self.cropping,self.cropping,self.width-2*self.cropping,self.height-2*self.cropping
        return window


    def _float_window(self,window):
        if window and self.float_cropping:
            x=window[0]+self.float_x
            y=window[1]+self.float_y
            w=window[2]-2*self.float_cropping
            h=window[3]-2*self.float_cropping
            window=x,y,w,h
        return window


    def _set_dimensions(self,path):
        if not (self.width and self.height):
            tmp,_=io.read(path)
            self.height,self.width=tmp.shape[1:]


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




#
# TILLER
#
class Tiller(object):
    """ Tiller
    
    For a given boundary shape generate windows of a given size and overlap

    Args:
        boundary_width/height<int|None>: 
            - width/height of boundary
            - required if boundary shape not specified
        boundary_shape<tuple|None>:
            - shape tuple
            - required if width/height not specified
        size<int>: tile size (width/height - only supports square tiles)
        overlap<int>: overlap between tiles
    """
    def __init__(
            self,
            boundary_width=None,
            boundary_height=None,
            boundary_shape=None,
            size=DEFAULT_SIZE,
            overlap=DEFAULT_OVERLAP):
        if not overlap:
            overlap=0
        self.size=size
        self.inner_size=self.size-2*overlap
        self._set_shape_attributes(
            boundary_width,
            boundary_height,
            boundary_shape,
            overlap)
        self.length=self.cols*self.rows
            
            
    def column_row(self,index):
        col=index//self.rows
        row=index-col*self.rows
        return int(col), int(row)
    

    def window(self,index=None,col=None,row=None):
        if index is not None:
            col,row=self.column_row(index)
        self.col,self.row=col,row  
        col_off=self.col_offset+col*self.size
        row_off=self.row_offset+row*self.size
        return (col_off,row_off,self.size,self.size)
    

    def __len__(self):
        return self.length
            

    def __getitem__(self, index):
        if (index>=self.length):
            raise IndexError
        self.index=index
        return self.window(index=self.index)
            
            
    #
    # INTERNEL
    #
    def _set_shape_attributes(
            self,
            boundary_width,
            boundary_height,
            boundary_shape,
            overlap):
        if boundary_shape:
            boundary_height,boundary_width=boundary_shape
        self.cols=math.floor((boundary_width-2*overlap)/self.inner_size)
        self.rows=math.floor((boundary_height-2*overlap)/self.inner_size)
        self.width=2*overlap+self.cols*self.inner_size
        self.height=2*overlap+self.rows*self.inner_size
        self.col_offset=math.floor((boundary_width-self.width)/2)
        self.row_offset=math.floor((boundary_height-self.height)/2)
        self.overlap=overlap

