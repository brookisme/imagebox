import numpy as np
from imagebox.config import FIRST, LAST, BAND_ORDERING, BANDS_FIRST
#
# CONSTANTS
# 
EPS=1e-8


#****************************************************************
#
# IMPORTANT NOTE: band index definition based on band order
#                 red, green, blue, nir, red-edge, swir1
#
#****************************************************************

BANDS=[
    'red',
    'green',
    'blue',
    'nir',
    'red-edge',
    'swir1',
    'swir2'
]
INDICES={
    'ndvi':(3,0),
    'ndwi':(1,3),
    'ndwi_leaves':(3,5),
    'ndbi':(5,3),
    'built_up': ['ndbi','ndvi'],
    'greeness':{ 
        "numerator_bands":1,
        "denominator_bands":0,
    },
   'chlogreen':{
        "numerator_bands":3,
        "denominator_bands":[1,4],
    },
    'gcvi':{
        "numerator_bands":3,
        "denominator_bands":1,
        "constant":1
    },
    'evi_modis':{
        "numerator_bands":[3,0],
        "numerator_coefs":[2.5,-2.5],
        "denominator_bands":[3,0,2],
        "denominator_coefs":[1,6,7.5],
        "denominator_constant":1
    },
    'evi_s2':{
        "numerator_bands":[3,0],
        "numerator_coefs":[2.5,-2.5],
        "denominator_bands":[3,0,2],
        "denominator_coefs":[1,6,7.5],
        "denominator_constant":10000
    }
}


ORDERED='ordered'
ORDERD_BANDS=[
    'coastal-aerosol',
    'blue',
    'green',
    'red',
    'red-edge',
    'red-edge-2',
    'red-edge-3',
    'nir',
    'red-edge-4',
    'water-vapor',
    'cirrus',
    'swir1',
    'swir2'
]
INDICES_ORDERED={
    'ndvi':(7,3),
    'ndwi':(2,7),
    'ndwi_leaves':(7,11),
    'ndbi':(11,7),
    'built_up': ['ndbi','ndvi'],
}

S2_1020='s2_1020'
S2_1020_BANDS=[
    'B2',
    'B3',
    'B4',
    'B5',
    'B6',
    'B7',
    'B8',
    'B11',
    'B12'
]
INDICES_S2_1020={
    'ndvi':(6,2),
    'ndwi':(1,6),
    'ndwi_leaves':(6,8),
    'ndbi':(8,6),
    'built_up': ['ndbi','ndvi'],
}


#
# METHODS
#
def index(im,index_name,indices=None):
    """ band index based on name, or ndiff args or ratio-index args
    
    Args:
        im<np.array>: image array
        index_name<str|False>: index key from INDICES or False for custom index
    """
    if isinstance(indices,str):
        if indices==ORDERED:
            indices=INDICES_ORDERED
        elif indices==S2_1020:
            indices=INDICES_S2_1020
        else:
            indices=INDICES
    elif not indices: 
        indices=INDICES
    args=indices[index_name]
    if isinstance(args,dict):
        return ratio_index(im,**args)
    else:
        if isinstance(args[0],str):
            a=index(im,args[0])
            b=index(im,args[1])
            return a-b
        else:
            return normalized_difference(im,*args)


def normalized_difference(im,band_1,band_2,bands_first=BANDS_FIRST):
    """ Normalize Difference

        Args:
            im<np.array>: image array
            b1<int>: the band number for b1
            b2<int>: the band number for b2  


        Returns:
            <arr>: (b1-b2)/(b1+b2)
    """
    im=im.astype(np.float)
    if bands_first:
        band_1=im[band_1]
        band_2=im[band_2]
    else:
        band_1=im[:,:,band_1]
        band_2=im[:,:,band_2]
    return np.divide(band_1-band_2,band_1+band_2+EPS)


def ratio_index(
        im,
        numerator_bands,
        denominator_bands=None,
        numerator_coefs=None,
        denominator_coefs=None,
        numerator_constant=0,
        denominator_constant=0,
        constant=0):
    """ Ratio Index
        
        Generalized Index that allows for any linear combination of bands
        in numerator and denominator plus an overall constant.

    """
    if not constant:
        constant=0
    numerator=linear_combo(
        im,
        bands=numerator_bands,
        coefs=numerator_coefs,
        constant=numerator_constant)
    if denominator_bands is None:
        denominator=1
    else:  
        denominator=linear_combo(
            im,
            bands=denominator_bands,
            coefs=denominator_coefs,
            constant=denominator_constant)
    return np.divide(numerator,denominator+EPS)+constant


def linear_combo(im,bands,coefs=None,constant=None,bands_first=BANDS_FIRST):
    """ Linear Combination

        Args:
            im<np.array>: image array   
            bands <list|int>: list of band indices or band index
            coefs <list|int|None>: list of coefs, a coef for all bands, or None -> 1
            constant <float[0]>: additive constant
        Returns:
            <np.array>: image_bands dot coefs + constant
    """
    im=im.astype(np.float)
    if not constant:
        constant=0
    if isinstance(bands,int):
        bands=[bands]
    if not coefs:
        coefs=1
    if isinstance(coefs,int):
        coefs=[coefs]*len(bands)
    if bands_first:
        im=coefs[0]*im[bands[0]]
        for c,b in zip(coefs[1:],bands[1:]):
            im+=c*im[b]
    else:
        im=coefs[0]*im[:,:,bands[0]]
        for c,b in zip(coefs[1:],bands[1:]):
            im+=c*im[:,:,b]
    return im+constant


def shadow_mask(im,band_bounds=[77,77,87],max_diff=25,bands=[0,1,2],blueness=6):
    bbnds=[]
    for b,bnd in zip(bands,band_bounds):
        bbnds.append(im[b]<=bnd)
    bbnds=np.array(bbnds)
    isblue=im[bands[:-1]].max(axis=0)+blueness<=im[bands[-1]]
    isgrey=(im[bands[-1]]-im[bands[:-1]].min(axis=0))<=max_diff
    return bbnds.all(axis=0)*isblue*isgrey

