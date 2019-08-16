import os
import random
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from affine import Affine
#
# CONSTANTS
#
FIRST='first'
RESAMPLING=Resampling.bilinear


#
# READ/WRITE
#
def read(path,window=None,window_profile=True,return_profile=True,dtype=None):
    """ read image
    Args: 
        - path<str>: source path
        - window<tuple|Window>: col_off, row_off, width, height
        - window_profile<bool>:
            - if True return profile for the window data
            - else return profile for the src-image
        - dtype<str>:
    Returns:
        <tuple> np.array, image-profile
    """
    with rio.open(path,'r') as src:
        if return_profile:
            profile=src.profile
        if window:
            image=src.read(window=Window(*window))
            if window_profile and return_profile:
                profile=update_profile(profile,window=window)  
        else:
            image=src.read()
        if dtype:
            image=image.astype(dtype)
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
        ims=[scale_read(p,r/stack_res) for p,r in zip(paths,res_list)]
    else:
        ims=[read(p,return_profile=False) for p in paths]

    return np.concatenate(ims), profile


def scale_read(path,scale=1,resampling=RESAMPLING,band=1):
    """ open and rescale image
    Args: 
        - path<str>: source path
        - scale<int|float>: amount to scale image by
        - resampling<str>: resampling method
        - band<int|None>: band to read
    """
    with rio.open(path) as src:
        if scale==1:
            im=src.read(indexes=band)
        else:
            im=src.read(
                indexes=band,
                out_shape=(int(src.height*scale),int(src.width*scale)),
                resampling=resampling)
    return im
        
        
        



#
# WINDOW/PROFILE HELPERS
#
def update_profile(
        profile,
        col_off=None,
        row_off=None,
        width=None,
        height=None,
        window=None):
    """ new profile based on original profile and window """
    if window:
        col_off, row_off, width, height=window
    affine=profile['transform']
    res=affine.a
    x0=affine.c
    y0=affine.f
    deltax=col_off*res
    deltay=row_off*res
    xmin=x0+deltax
    ymin=y0-deltay
    affine=Affine(res, 0.0, xmin,0.0, -res, ymin)
    profile=profile.copy()
    if width: profile['width']=width
    if height: profile['height']=height
    profile['transform']=affine
    profile['blockxsize']=min(width,profile.get('blockxsize',0))
    profile['blockysize']=min(height,profile.get('blockysize',0))
    return profile











