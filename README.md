### Image Kit: python utilities for working with multispectral imagery 

ImageKit contains four main modules:

- [io](#io): a rasterio wrapper for reading/writing imagery. Simplifies reading windows by returning a window specific profile
- [processor](#processor): a number of methods for processing images such as normalization, mapping categorical values, augmentation, etc.
- [indices](#indicies): simplifies computing band-indices. includes a number of preset band indices, such as NDVI, NDWI, BuiltUp-Index.
- [handler](#handler): A class that handles processing for target and input data simultaneously. This is particularly useful in machine-learning. The class simplifies the creation of (pytorch) Datasets/Dataloaders or (keras) data-generators.
 

---

<a name='install'></a>
##### INSTALL

```bash
git clone https://github.com/brookisme/image_kit.git
cd image_kit
pip install -e .
```

---

<a name='io'></a>
##### IO

_a rasterio wrapper for reading/writing imagery_

This module contains two simple methods:

##### io.read(path,window=None,window_profile=True,dtype=None)

```python
Args: 
    - path<str>: source path
    - window<tuple|Window>: col_off, row_off, width, height
    - window_profile<bool>:
        - if True return profile for the window data
        - else return profile for the src-image
    - dtype<str>:
Returns:
    <tuple> np.array, image-profile
```

##### io.write(im,path,profile,makedirs=True) 

```python
    Args: 
        - im<np.array>: image
        - path<str>: destination path
        - profile<dict>: image profile
        - makedirs<bool>: if True create necessary directories
```


---

<a name='processor'></a>
##### Processor

This module contains a number of methods for processing images. See doc-strings for details. Here current list of methods:

- center: center image around mean
- normalize: normalize image
- denormalize: turn a normalized image into an RGB "denormalized" image
- map_values: map categorical pixel values to new values
- to_categorical: turn categorical image into a categorical (binary-multi-band) image
- crop: crop image
- augmentation: returns a random flip and/or rotation value to be used when augmenting data
- augment: augment data with flips and/or 90-degree rotations


---

<a name='indices'></a>
##### Indices

This module allows you to compute combination of band values to create band indices.  There are a number of pre-configured band combinations as well as methods for **normalized difference** band combinations, **linear combinations** of bands and **ratios of linear combinations** of bands.

IMPORTANT NOTE: band index definitions in `INDICES`  are based on band ordering red, green, blue, nir, red-edge, swir1.  If using different bands/band-ordering you can use `INDICES` as a guide to how one constructs band-indices.

Here is a list of pre-configured indices:

- ndvi
- ndwi
- ndwi_leaves
- ndbi
- built_up
- greeness
- chlogreen
- gcvi
- evi_modis
- evi_s2

examples:

```python
# NDVI from predefined structure
ndvi=indices.index(im,'ndvi')

# NDVI from normalized_difference method:
ndvi2=indices.normalized_difference(im,3,0)
```

Those are the simplest examples, but you should be able to create almost any combination of the bands using this module. See doc-strings for details. Here current list of methods:

- index: handles pre-configured indices and is a wrapper method for all methods below
- normalized_difference: for bands b1,b2 computes `(b1-b2)/(b1+b2)`
- linear_combo: for bands b1,...bN and constant C computes `b1+b2+...+bN + C`
- ratio_index: for bands n1,...,nN and d1,...,dM and constants C, Cn, Cd computes `((n1+n2+...+nN + Cn)/(d1+d2+...+dN + Cd))+C`

---

<a name='handler'></a>
##### Handler

_handlers processing for target and input data_

This module contains two classes:

- [InputTargetHandler](#inpttarg): Processes input and target data in conjunction
- [Tiller](#tiller): for a given boundary shape, this generates windows of a given size and overlap which can be used to tile an image

<a name='inpttarg'>

###### InputTargetHandler

The InputTargetHandler is able to:

- compute band indices (ndvi, ndwi, ...)
- normalize or center the input imagery
- augment data (flip/90-deg-rotation)
- map the values in the target imagery to new values
- convert the target to a categorical (binary-multi-band) image
- select bands
- crop input and/or target data
- tile the input/target into a grid of images (i.e. a single 900x900 image can be treated as 9 300x300 images)
- (float_cropping) for a window size smaller than the image (or image tile) randomly selecting a window at a arbitrary point within the the image (or image tile)

An example speaks some number of words:

The example below creates a pytorch Dataloader/Dataset from a dataframe with rows-containing input and target filenames. The bulk of the code is simply getting and returning those input and target filenames. All the manipulation is done by InputTargetHandler:

```python
class UrbanLandUseDS(Dataset):
    @staticmethod
    def load_dataframe(dataframe):
        if isinstance(dataframe,str):
            dataframe=pd.read_csv(dataframe)
        return dataframe


    @classmethod
    def loader(cls,
            dataframe,
            batch_size=DEFAULT_BATCH_SIZE,
            partial_batches=False,
            loader_kwargs={},
            **kwargs):
        r""" convenience method for loading the DataLoader directly.
            
            Args:
                see class args

            Returns:
                dataloader 
        """


        return DataLoader(cls(dataframe,**kwargs),batch_size=batch_size,**loader_kwargs)



    def __init__(self,
            dataframe,
            data_dir=DATA,
            resolution=RESOLUTION,
            input_bands=None,
            means=None,
            stdevs=None,
            band_indices=None,
            value_map=VALUE_MAP,
            default_mapped_value=NB_CATEGORIES,
            to_categorical=False,
            nb_categories=NB_CATEGORIES,
            augment=True,
            cropping=None,
            float_cropping=None,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE,
            randomize=True,
            train_mode=True):
        self.randomize=randomize
        self._set_data(dataframe)
        self.train_mode=train_mode
        self.root_dir=f'{data_dir}/{resolution}'
        self.handler=InputTargetHandler(
            input_bands=input_bands,
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            value_map=value_map,
            default_mapped_value=default_mapped_value,
            to_categorical=to_categorical,
            nb_categories=nb_categories,
            augment=augment,
            cropping=cropping,
            float_cropping=float_cropping,
            input_dtype=input_dtype,
            target_dtype=target_dtype)


    def __len__(self):
        return len(self.aoi_names)


    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_float_window()
        self.handler.set_augmentation()
        inpt,inpt_p=self.handler.input(self.input_path,return_profile=True)
        targ,targ_p=self.handler.target(self.target_path,return_profile=True)
        inpt_p=self._clean(inpt_p)
        targ_p=self._clean(targ_p)
        if self.train_mode:
            itm={
                'input': inpt, 
                'target': targ }
        else:
            itm={
                'input': inpt, 
                'target': targ,
                'index': self.index,
                'aoi_name': self.aoi_name,
                'input_path': self.input_path,
                'target_path': self.target_path,
                'float_x': self.handler.float_x,
                'float_y': self.handler.float_y,
                'k': self.handler.k,
                'flip': self.handler.flip,
                'input_profile': inpt_p,
                'target_profile': targ_p }
        return itm
            

    def select_data(self,index):
        """ select data for index without loading/processing images
        """
        self.index=index
        self.aoi_name=self.aoi_names[index]
        self.row=self.dataframe[self.dataframe.aoi_name==self.aoi_name].sample().iloc[0]
        self.target_path=f'{self.root_dir}/target/{self.row.region}/{self.row.target_file}'
        self.input_path=f'{self.root_dir}/input/{self.row.region}/{self.row.input_file}'


    def reset(self,limit=None):
        """ reset the generator
            * if randomize: shuffle aoi
            * if limit: limit aois, reset size
        """
        if self.randomize:
            shuffle(self.aoi_names)
        if limit:
            self.aoi_names=self.aoi_names[:limit]
            self.size=len(self.aoi_names)

    #
    # INTERNAL
    #
    def _set_data(self,dataframe):
        self.dataframe=UrbanLandUseDS.load_dataframe(dataframe)
        self.aoi_names=list(self.dataframe.aoi_name.unique())
        self.size=len(self.aoi_names)
        self.reset()
        

    def _clean(self,obj):
        return  { k:v for k,v in obj.items() if v is not None }

```


<a name='tiller'>
    
###### Tiller

For a given boundary shape generate windows (x-offset, y-offset, width, height) of a given size and overlap

```
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
```


