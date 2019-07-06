import os
import random
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from affine import Affine
from torch_kit.utils.image_processor import normalize, center


#
# READ/WRITE
#
def read(path,window=None,window_profile=True,dtype=None):
    """ read image
    """
    with rio.open(path,'r') as src:
        profile=src.profile
        if window:
            image=src.read(window=Window(*window))
            if window_profile:
                profile=update_profile(profile,window=window)  
        else:
            image=src.read()
        if dtype:
            image=image.astype(dtype)
    return image, profile



def write(im,path,profile,makedirs=True):
    """ write image
    Args: 
        - im (np.array): image
        - path (str): destination path
        - profile (dict): image profile
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











