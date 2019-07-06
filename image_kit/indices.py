import numpy as np
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

# BU = NDBI - NDVI


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



#
# METHODS
#
def index(im,index,*normalized_difference_args,**ratio_index_kwargs):
    """ band index based on name, or ndiff args or ratio-index args
    
    Args:
        im<np.array>: image array
        index<str|False>: index key from INDICES or False for custom index
        *normalized_difference_args: args for normalized_difference
        **ratio_index_kwargs: config or ratio_index
    """
    if index:
        args=INDICES[index]
    else:
        if normalized_difference_args:
            args=normalized_difference_args
        else:
            args=ratio_index_kwargs
    if isinstance(args,dict)
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



