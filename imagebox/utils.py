import rasterio.transform as transform
from rasterio.crs import CRS
from imagebox.config import FIRST, LAST, BAND_ORDERING



def get_crs(crs,as_dict=False):
    if isinstance(crs,int):
        crs_dict={'init':f'epsg:{crs}'}
    elif isinstance(crs,str):
        crs_dict={'init':crs}
    else:
        return crs
    if as_dict:
        return crs_dict
    else:
        return CRS(crs_dict)


def crs_res_bounds(profile):
    """ get crs, resolution and bounds form image profile """
    affine=profile['transform']
    res=affine.a
    minx=affine.c
    miny=affine.f-profile['height']*res
    maxx=minx+profile['width']*res
    maxy=miny+profile['height']*res
    crs=str(profile['crs'])
    return crs,res,(minx,miny,maxx,maxy)



def bounds_from_profile(profile):
    """ bounds from profile """
    a=profile['transform']
    h=profile['height']
    w=profile['width']
    return transform.array_bounds(h,w,a)


def profile(
        crs,
        transform,
        width=None,
        height=None,
        size=None,
        count=1,
        nodata=None,
        dtype='uint8',
        compress='lzw'):
    """ construct profile """
    if size:
        width=height=size
    return {
        'crs': get_crs(crs),
        'transform': transform,
        'width': width,
        'height': height,
        'count': count,
        'nodata': nodata,
        'dtype': dtype,
        'compress': compress,
        'driver': 'GTiff',
        'interleave': 'pixel' }


#
# IO
#
def order_bands(image,band_ordering=None):
    if band_ordering is None:
        band_ordering=BAND_ORDERING 
    if band_ordering.lower()==LAST:
        image=image.transpose(1,2,0)
    return image


#
# WINDOW HELPERS
#
def window_origin(src_transform,target_transform):
    """ origin of window from src and window affine-transform """
    x=(target_transform.c-src_transform.c)/target_transform.a
    y=(target_transform.f-src_transform.f)/target_transform.e
    return round(x),round(y)


def profiles_to_window(src_profile,target_profile):
    """ pixel window from src and window profile """
    x,y=window_origin(src_profile['transform'],target_profile['transform'])
    return Window(x,y,target_profile['width'],target_profile['height'])


def crop_src_by_target_profile(src,target_profile):
    im=src.read()
    window=profiles_to_window(src.profile,target_profile)
    imin=window.row_off
    jmin=window.col_off
    imax=imin+window.width
    jmax=jmin+window.height
    return im[:,imin:imax,jmin:jmax]



