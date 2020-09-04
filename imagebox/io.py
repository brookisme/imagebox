import os
import random
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from affine import Affine
from imagebox.config import FIRST, LAST, BAND_ORDERING
from . import utils
#
# CONSTANTS
#
RESAMPLING=Resampling.bilinear

#
# READ/WRITE
#
def read(
        path,
        window=None,
        window_profile=True,
        return_profile=True,
        res=None,
        scale=None,
        out_shape=None,
        bands=None,
        resampling=RESAMPLING,
        band_ordering=None,
        dtype=None):
    """ read image
    Args: 
        - path<str>: source path
        - window<tuple|Window>: col_off, row_off, width, height
        - window_profile<bool>:
            - if True return profile for the window data
            - else return profile for the src-image
        - res<int>: rescale to new resolution. overides scale and out_shape
        - scale<float>: rescale image res=>res*scale overrides out_shape
        - out_shape<tuple>: (h,w) rescales image. overwritten by res and scale
        - dtype<str>:
    Returns:
        <tuple> np.array, image-profile
    """
    with rio.open(path,'r') as src:
        if return_profile:
            profile=src.profile
        if window:
            w,h=window[2], window[3]
            window=Window(*window)
            if window_profile and return_profile:
                profile['transform']=src.window_transform(window)
                profile['width']=w
                profile['height']=h
        else:
            w,h=src.width, src.height
        if res:
            scale=src.res[0]/res
        if scale:
            out_shape=(int(h*scale),int(w*scale))
        if out_shape and return_profile:
            profile=rescale_profile(profile,out_shape)
        image=src.read(
                indexes=bands,
                window=window,
                out_shape=out_shape,
                resampling=resampling )
        if dtype:
            image=image.astype(dtype)
        image=utils.order_bands(image,band_ordering)
    if return_profile:
        return image, profile
    else:
        return image


def write(im,path,profile,makedirs=True):
    """ write image
    Args: 
        - im<np.array>: image
        - path<str>: destination path
        - profile<dict>: image profile
        - makedirs<bool>: if True create necessary directories
    """  
    if makedirs:
        dirname=os.path.dirname(path)
        if dirname:
            os.makedirs(os.path.dirname(path),exist_ok=True)
    affine_transform=profile.get('affine')
    if affine_transform:
        profile['transform']=affine_transform
    with rio.open(path,'w',**profile) as dst:
        dst.write(im)
        

def read_stack(paths,res_list=None,stack_res=FIRST,resampling=RESAMPLING):
    """ band-wise read for images

    Args: 
        - paths<list>: list of source paths (per-band/ordered)
        - res_list<list|None>: 
            * list of resolutions.  
            * if passed will rescale bands to stack_res
        - stack_res:
            * resolution to rescale all bands to
            * if 'first' use the first resolution in res_list
        - resampling<str>: resampling method
    """
    with rio.open(paths[0]) as src:
        profile = src.profile
    profile.update(count=len(paths))
    if res_list:
        if isinstance(res_list,int):
            res_list=[res_list]*len(paths)
        if stack_res is FIRST:
            stack_res=res_list[0]
        scale=(stack_res/res_list[0])
        if scale!=1.0:
            profile['width']*=scale
            profile['height']*=scale
        ims=[ read(p,scale=r/stack_res,band=1,resampling=resampling) 
              for p,r in zip(paths,res_list) ]
    else:
        ims=[read(p,return_profile=False) for p in paths]

    return np.concatenate(ims), profile        


#
# WINDOW/PROFILE HELPERS
#
def rescale_profile(profile,out_shape):
    affine=profile['transform']
    h_out,w_out=out_shape
    h,w=profile['height'],profile['width']
    res_y=int(affine.e*h/h_out)
    res_x=int(affine.a*w/w_out)
    profile['transform']=Affine(res_x, 0.0, affine.c,0.0, res_y, affine.f)
    profile['height'],profile['width']=h_out,w_out
    return profile







