import math
from random import randint
from rasterio.enums import Resampling
import numpy as np
import gcs_helpers.fetch as gfetch
import imagebox.io as io
import imagebox.processor as proc
import imagebox.indices as indices
from imagebox.config import FIRST, LAST, BAND_ORDERING, BANDS_FIRST

#
# CONSTANTS
# 
INPUT_DTYPE=np.float
TARGET_DTYPE=np.int64
DEFAULT_SIZE=256
DEFAULT_OVERLAP=0
INPUT_RESAMPLING=Resampling.bilinear
TARGET_RESAMPLING=Resampling.mode
TO_CATEGORICAL_ERROR=(
    'imagebox.handler: '
    'nb_categories required for to_categorical' )
SAFE_RESCALE_ERROR=(
    'imagebox.handler: '
    'target rescale leads to fractional value. '
    'use safe_rescale=False to force' )
DIMS_REQUIRED_ERROR=(
    'imagebox.handler: '
    'width and height, or example_path, required for (float)cropping' )

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
            - list of arguments for imagebox.indices.index
            - adds band_index bands to image
        indices_dict<dict>:
            - preset indices from which to select band_indices
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
        flip_target/input<bool>: 
            - flip target/input image before any additional augmentation 
            - for use when target and input pairs have opposite y-axis orientation
            - set only one of these to be true
        target/input_resolution: rescale input or target with input/target resampling method
        input/target_resampling: resampling method for input or target
        padding|input/target_padding<int|None>: 
            - amount to pad images|input/target-image after processing the image
            - will be ignored if there is cropping
        cropping<int|None>:
            - amount to crop both input and target image when reading the image
        input/target_cropping<int|None>: 
            - amount to crop input/target image after processing the image
        float_cropping<int|float|None>: 
            - remove pixels from h/w of input and target starting at random i,j.
            - 2*float_cropping must be int
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
            indices_dict=None,
            value_map=None,
            default_mapped_value=proc.DEFAULT_VMAP_VALUE,
            to_categorical=False,
            nb_categories=None,
            augment=True,
            flip_target=False,
            flip_input=False,
            input_resolution=None,
            target_resolution=None,
            input_bounds=None,
            input_resampling=INPUT_RESAMPLING,
            target_resampling=TARGET_RESAMPLING,
            padding=None,
            input_padding=None,
            target_padding=None,
            input_padding_value=0,
            target_padding_value=0,
            stop_floating=False,
            float_cropping=None,           
            target_ratio=1,
            safe_rescale=True,
            cropping=None,
            input_cropping=None,
            target_cropping='auto',
            window_index=None,
            size=None,
            width=None,
            height=None,
            example_path=None,
            tiller=None,
            tiller_config={},
            target_expand_axis=None,
            input_preprocess=None,
            target_preprocess=None,
            target_squeeze=True,
            read_from_gcs=False,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE ):
        if tiller is True:
            self.tiller=Tiller(**tiller_config)
        else:
            self.tiller=tiller
        self.input_bands=input_bands
        self.band_indices=band_indices
        self.indices_dict=indices_dict
        self.value_map=value_map
        self.default_mapped_value=default_mapped_value
        self.means=means
        self.stdevs=stdevs
        if to_categorical and (not nb_categories):
            raise ValueError(TO_CATEGORICAL_ERROR)
        self.to_categorical=to_categorical
        self.nb_categories=nb_categories
        self.augment=augment
        self.flip_target=flip_target
        self.flip_input=flip_input
        self.input_resolution=input_resolution
        self.target_resolution=target_resolution
        self.input_bounds=input_bounds
        self.input_resampling=input_resampling
        self.target_resampling=target_resampling
        self.set_augmentation()
        self.input_padding=input_padding or padding
        self.target_padding=target_padding or padding
        self.input_padding_value=input_padding_value
        self.target_padding_value=target_padding_value
        self._set_cropping_and_dims(
            cropping,
            target_ratio,
            safe_rescale,
            input_cropping,
            target_cropping,
            float_cropping,
            stop_floating,
            size,
            width,
            height,
            example_path)
        self.set_window(window_index=window_index)
        self.target_expand_axis=target_expand_axis
        self.input_preprocess=input_preprocess
        self.target_preprocess=target_preprocess
        self.target_squeeze=target_squeeze
        self.read_from_gcs=read_from_gcs
        self.input_dtype=input_dtype
        self.target_dtype=target_dtype


    def input(self,path,window=None,return_profile=False):
        self.input_path=path
        im,profile=self._read(
            path,
            self.input_resolution,
            self.target_resampling,
            window or self.input_window )
        im=process_input(
            im,
            preprocess=self.input_preprocess,
            flip=self.flip_input,
            input_bands=self.input_bands,
            band_indices=self.band_indices,
            indices_dict=self.indices_dict,
            padding=self.input_padding,
            padding_value=self.input_padding_value,
            bounds=self.input_bounds,
            means=self.means,
            stdevs=self.stdevs,
            dtype=self.input_dtype )
        return self._return_data(
            im,
            profile,
            return_profile )


    def target(self,path,window=None,return_profile=False):
        self.target_path=path
        im,profile=self._read(
            path,
            self.target_resolution,
            self.target_resampling,
            window or self.target_window )
        im=process_target(
            im,
            preprocess=self.target_preprocess,
            flip=self.flip_target,
            value_map=self.value_map,
            default_mapped_value=self.default_mapped_value,
            categorical=self.to_categorical,
            nb_categories=self.nb_categories,
            padding=self.target_padding,
            padding_value=self.target_padding_value,
            expand_axis=self.target_expand_axis,
            squeeze=self.target_squeeze,
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
    

    def set_window(self,window=None,window_index=None,example_path=None):
        if window:
            input_window=window
            target_window=window
        elif self.tiller:
            if window_index is None:
                window_index=randint(0,len(self.tiller)-1)
            self.window_index=window_index            
            target_window=self.tiller[self.window_index]
            input_window=target_window
        else:
            self.window_index=False
            input_window=0,0,self.input_width,self.input_height
            target_window=0,0,self.target_width,self.target_height
        self.input_window=self._shift_crop_window(
                input_window,
                dx=self.input_cropping,
                dy=self.input_cropping,
                crop=self.input_cropping)
        self.target_window=self._shift_crop_window(
                target_window,
                dx=self.target_cropping,
                dy=self.target_cropping,
                crop=self.target_cropping)
        if self.float_cropping:
            self.float_x=self._random_delta()
            self.float_y=self._random_delta()
            self.input_window=self._shift_crop_window(
                    self.input_window,
                    dx=self.float_x,
                    dy=self.float_y,
                    crop=self.float_cropping )
            self.target_window=self._shift_crop_window(
                    self.target_window,
                    dx=self._target_rescale(self.float_x),
                    dy=self._target_rescale(self.float_y),
                    crop=self._target_rescale(self.float_cropping) )


    
    #
    # INTERNAL METHODS
    #
    def _set_cropping_and_dims(self,
            cropping,
            target_ratio,
            safe_rescale,
            input_cropping,
            target_cropping,
            float_cropping,
            stop_floating,
            size,
            width,
            height,
            example_path):
        self.target_ratio=target_ratio or 1
        self.input_cropping=input_cropping or cropping or 0
        float_cropping=float_cropping or 0
        if not float(2*float_cropping).is_integer():
            raise ValueError('`2*float_cropping` must be int')
        self.safe_rescale=safe_rescale
        if size:
            self.input_width=size
            self.input_height=size
        else:
            self.input_width=width
            self.input_height=height
        self._ensure_dimensions(example_path)
        self.target_width=self._target_rescale(self.input_width)
        self.target_height=self._target_rescale(self.input_height)
        if stop_floating:
            self.input_cropping+=float_cropping
            self.float_cropping=False
        else:
            self.float_cropping=float_cropping
        if self.input_cropping and (target_cropping=='auto'):
            self.target_cropping=self._target_rescale(self.input_cropping)
        elif isinstance(target_cropping,(int,float)):
            self.target_cropping=target_cropping
        else:
            self.target_cropping=None


    def _target_rescale(self,value):
        if self.target_ratio==1:
            return value
        else:
            fval=value*self.target_ratio
            val=round(fval)
            if (not self.safe_rescale) or (val==fval):
                return val
            else:
                raise ValueError(SAFE_RESCALE_ERROR)


    def _read(self,path,resolution,resampling,window):
        if self.read_from_gcs:
            im,p=gfetch.image(
                path=path,
                window=window,
                res=resolution,
                resampling=resampling,
                band_ordering=BAND_ORDERING)
        else:
            im,p=io.read(
                path,
                window=window,
                res=resolution,
                resampling=resampling)
        return im,p


    def _random_delta(self):
        return randint(0,2*int(self.float_cropping*self.target_ratio))


    def _shift_crop_window(self,window,dx=0,dy=0,crop=0):
        if dx or dy or crop:
            x=window[0]+dx
            y=window[1]+dy
            w=window[2]-2*crop
            h=window[3]-2*crop
            return int(x),int(y),int(w),int(h)
        else:
            return window


    def _ensure_dimensions(self,example_path):
        if not (self.input_width and self.input_height):
            if example_path:
                shape=io.read(example_path,return_profile=False).shape
                if BANDS_FIRST:
                    self.input_height,self.input_width=shape[1:]
                else:
                    self.input_height,self.input_width=shape[:2]
            elif self.input_cropping or self.target_cropping or self.float_cropping:
                raise ValueError(DIMS_REQUIRED_ERROR)


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
        rotate=False,
        flip=False,     
        input_bands=None,
        band_indices=None,
        indices_dict=None,
        padding=None,
        padding_value=0,
        cropping=None,
        bounds=None,
        means=None,
        stdevs=None,
        preprocess=None,
        dtype=INPUT_DTYPE):
    if preprocess:
        im=preprocess(im)
    im=proc.augment(im,k=rotate,flip=flip)
    if cropping:
        im=proc.crop(im,cropping)
    if band_indices:
        index_bands=[indices.index(im,idx,indices_dict) for idx in band_indices]
    if means is not None:
        if stdevs is None:
            im=proc.center(im,means=means,to_int=False)
        else:
            im=proc.normalize(im,means=means,stdevs=stdevs)
    if input_bands:
        if BANDS_FIRST:
            im=im[input_bands]
        else:
            im=im[:,:,input_bands]
    if band_indices:
        if BANDS_FIRST:
            if input_bands is False:
                im=np.vstack([index_bands])
            else:
                im=np.vstack([im,index_bands])
        else:
            if input_bands is False:
                im=np.dstack(index_bands)
            else:
                im=np.dstack([im]+index_bands)
    if (not cropping) and padding:
        im=proc.pad(im,padding=padding,value=padding_value)
    if bounds:
        for i,b in bounds.items():
            i=int(i)
            im[i]=im[i].clip(min=b.get('min'),max=b.get('max'))
    return im.astype(dtype)


def process_target(
        im,
        rotate=False,
        flip=False,
        value_map=None,
        default_mapped_value=proc.DEFAULT_VMAP_VALUE,
        categorical=False,
        nb_categories=None,
        padding=None,
        padding_value=0,
        cropping=None,
        expand_axis=None,
        preprocess=None,
        squeeze=True,
        dtype=TARGET_DTYPE):
    if preprocess:
        im=preprocess(im)
    im=proc.augment(im,k=rotate,flip=flip)
    if squeeze:
        im=np.squeeze(im)
    if value_map:
        im=proc.map_values(im,value_map)
    if categorical:
        im=proc.to_categorical(im,nb_categories)
    if cropping:
        im=proc.crop(im,cropping)
    elif padding:
        im=proc.pad(im,padding=padding,value=padding_value)
    if expand_axis is not None:
        if expand_axis is True:
            expand_axis=0
        im=np.expand_dims(im,axis=expand_axis)
    return im.astype(dtype)




#
# TILLER
#
class Tiller(object):
    """ Tiller
    
    For a given boundary shape generate windows (x-offset, y-offset, width, height) of a given size and overlap

    Usage:
        im=np.arange(1024**2).reshape((1024,1024))
        tiller=hand.Tiller(boundary_shape=im.shape,size=100,overlap=10)
        xoff,yoff,width,height=tiller[0]
        print("NB WINDOWS:",len(tiller))
        print("WINDOW-0:",xoff,yoff,width,height)
        im[yoff:yoff+height,xoff:xoff+width]
        ### output:
        NB WINDOWS: 115600
        WINDOW-0: 1 1 5 5
        array([[1025, 1026, 1027, 1028, 1029],
               [2049, 2050, 2051, 2052, 2053],
               [3073, 3074, 3075, 3076, 3077],
               [4097, 4098, 4099, 4100, 4101],
               [5121, 5122, 5123, 5124, 5125]])

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

