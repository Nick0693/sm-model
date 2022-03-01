import os
import glob
from functools import reduce

import yaml
import pandas as pd
import ee

from timeout import timeout

ee.Initialize()

"""
Downloads and converts remotely sensed point observations from Google Earth Engine
to a dataframe format. This includes cloud-masking Sentinel-2 data.
The cloud masking is developed by the Google team.

The Earth Engine (ee) package requires some steps to become operational in this workflow.
I suggest following the guide found here to get started:
https://developers.google.com/earth-engine/guides/python_install-conda

All configurable data can be found in the associated settings.yml file.
"""

### Cloud masking steps for Sentinel-2 images ###
## Creates an image collection from specified parameters of cloud probability
def get_s2_sr_cld_col(start_date, end_date, cloud_filter=80):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


## Cloud component
## Adds the s2cloudless probability layers and derives a cloud mask
def add_cloud_bands(img, cloud_probability_threshold=50):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(cloud_probability_threshold).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))


## Cloud shadow component
## Adds dark pixels, cloud projection, and identified shadows as bands to the image collection
def add_shadow_bands(img, nir_threshold=0.15, cloud_distance=1):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(nir_threshold*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, cloud_distance*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


## Assembles the cloud-shadow mask to produce a final masking of the images
def add_cld_shdw_mask(img):
    buffer = 50
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(buffer*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return img.addBands(is_cld_shdw)


## Converts an EarthEngine image collection into a Pandas array for a given point
## with the specified bands.
## The buffer attribute should be equal to half the spatial resolution of the final product.
## The bands should be given as a list, even for single bands.
@timeout(600) # limits the time allowed for the function to run. Increase the number if running on a slow network
def ee_to_df(ee_arr, lon, lat, buffer, int_limit, bands, start_date, end_date):
    # Converts columns to numeric values
    def to_numeric(dataframe, band):
        dataframe[band] = pd.to_numeric(dataframe[band], errors='coerce')
        return dataframe

    # Transform the client-side data to a dataframe
    poi = ee.Geometry.Point(lon, lat)
    try:
        arr = ee_arr.select(bands).getRegion(poi, 50).getInfo()
        df = pd.DataFrame(arr)
        headers = df.iloc[0]
        df = pd.DataFrame(df.values[1:], columns=headers)

        # Applies the to_numeric function and fills NaN rows with interpolated values
        for band in bands:
            df = to_numeric(df, band)
            if int_limit > 0:
                df[band].interpolate(method='linear', limit=int_limit, limit_direction='both',
                                     inplace=True)
            df.drop_duplicates(keep='first') # remove duplicates

        # Creates an index date column and drops unnecessary date, time, and coordinate columns
        df['Date'] = pd.to_datetime(df['time'], unit='ms')
        df['Date'] = df['Date'].dt.date
        df.set_index('Date', inplace=True)
        df.drop(['id', 'time', 'longitude', 'latitude'], axis=1, inplace=True)

    # Not ideal but I can't seem to catch the specific HttpError/EEException
    except:
        print(f'    No bands in collections: {bands}')
        df = pd.DataFrame(columns=bands)

    # Drop duplicate entries from the index and reindex the dataset to daily timesteps
    df = df[~df.index.duplicated()]
    df = df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
    df.index.name = 'Date'

    return df

def dry_days(df):
    dry_streak = []
    counter = 0
    for day, value in enumerate(df['P']):
        if value > 0:
            dry_streak.append(0)
            counter = 0
        else:
            counter += 1
            dry_streak.append(counter)
    df['DD'] = dry_streak
    df['DD'] = df['DD'].astype('float')
    return df

## Calculates and adds the vegetation index profiles from the S2 data.
## The indices are interpolated linearly 30 days in both directions.
## NDWI is based on https://doi.org/10.1016/S0034-4257(96)00067-3
def add_vegetation_index(df, name, band_1, band_2, interpolate_index=True):
    df[name] = (df[band_1] - df[band_2]) / (df[band_1] + df[band_2])
    if interpolate_index:
        df[name].interpolate(method='linear', limit=30, limit_direction='both', inplace=True)
    return df


def assemble_variables(lon, lat, start_date, end_date, variable_list,
                       point_id, settings):
    dataframes = []

    # MODIS daily land surface temperature
    # https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD11A1
    # Temperature is given in Kelvin with a scaling factor of 0.02
    if 'TS' in variable_list:
        modis = (ee.ImageCollection("MODIS/006/MOD11A1")
                  .filterDate(start_date, end_date)
                  .select('LST_Day_1km'))
        lst = ee_to_df(modis, lon, lat, 5, 5, ['LST_Day_1km'], start_date, end_date)
        lst = lst * 0.02 - 273.15 # scaling factor and Kelvin to Celcius conversion
        lst.rename(columns={'LST_Day_1km' : 'TS'}, inplace=True)
        dataframes.append(lst)

    # Sentinel-1 GRD C-band SAR data
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
    # Values are expressed in decibel on a logarithmic scale
    if 'VV' in variable_list or 'VH' in variable_list:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterDate(ee.Date(start_date), ee.Date(end_date))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
        if 'VV' in variable_list:
            try:
                s1_vv = ee_to_df(s1, lon, lat, 5, 0, ['VV'], start_date, end_date)
                dataframes.append(s1_vv)
            except:
                print('    Connection timed out on variable: VV')
        if 'VH' in variable_list:
            try:
                s1_vh = ee_to_df(s1, lon, lat, 5, 0, ['VH'], start_date, end_date)
                dataframes.append(s1_vh)
            except:
                print('    Connection timed out on variable: VH')

    # Sentinel-2 Level-2A surface reflectance
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR
    if 'NDVI' in variable_list or 'NDWI' in variable_list:
        s2_sr_cld_col_eval = get_s2_sr_cld_col(start_date, end_date)
        s2 = s2_sr_cld_col_eval.map(add_cld_shdw_mask)

        if 'NDVI' in variable_list:
            s2_ndvi = ee_to_df(s2, lon, lat, 5, 0, ['B4', 'B8'], start_date, end_date)
            s2_ndvi = add_vegetation_index(
                s2_ndvi, 'NDVI', 'B8', 'B4', interpolate_index=settings['interpolate_index'])
            s2_ndvi.drop(['B4', 'B8'], axis=1, inplace=True)
            dataframes.append(s2_ndvi)
        if 'NDWI' in variable_list:
            s2_ndwi = ee_to_df(s2, lon, lat, 5, 0, ['B8A', 'B11'], start_date, end_date)
            s2_ndwi = add_vegetation_index(
                s2_ndwi, 'NDWI', 'B8A', 'B11', interpolate_index=settings['interpolate_index'])
            s2_ndwi.drop(['B8A', 'B11'], axis=1, inplace=True)
            dataframes.append(s2_ndwi)

    # ERA-5 daily aggregate re-analysis data
    # https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY
    # Maximum temperature (air) uses Kelvin
    # Precipitation is in mm
    if 'TA' in variable_list or 'P' in variable_list:
        era5 = (ee.ImageCollection('ECMWF/ERA5/DAILY')
                .filterDate(start_date, end_date)
                .select(['mean_2m_air_temperature', 'total_precipitation']))
        if 'TA' in variable_list:
            era5_t = ee_to_df(era5, lon, lat, 5, 0, ['mean_2m_air_temperature'], start_date, end_date)
            era5_t['mean_2m_air_temperature'] = era5_t['mean_2m_air_temperature'] - 273.15
            era5_t.rename(columns={'mean_2m_air_temperature' : 'TA'}, inplace=True)
            dataframes.append(era5_t)
        if 'P' in variable_list:
            era5_p = ee_to_df(era5, lon, lat, 5, 0, ['total_precipitation'], start_date, end_date)
            era5_p['total_precipitation'] = era5_p['total_precipitation'] * 1000
            era5_p.rename(columns={'total_precipitation' : 'P'}, inplace=True)
            era5_p = dry_days(era5_p)
            dataframes.append(era5_p)

    # Merge dataframes
    try:
        df = reduce(lambda  left,right: pd.merge(left, right, on=['Date'], how='outer'), dataframes)
        df['SITE'] = point_id
    except TypeError:
        print('No variables specified.')

    return df
