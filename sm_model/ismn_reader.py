import os

import yaml
import numpy as np
import pandas as pd
from ismn.interface import ISMN_Interface

import plotting

"""
Reads and aggregates soil moisture and soil temperature from multiple sensors
in a single network that fits the ISMN standard.

This is not a downloader so data should be downloaded beforehand from here:
https://ismn.geo.tuwien.ac.at/en/

All configurable data can be found in the associated settings.yml file.
"""

def export(df, ismn_dict, settings):
    df.to_csv(os.path.join(settings['wrk_dir'], '{}_ISMN_Dataframe.csv'
                           .format(settings['project_name'])), index=True)

    yml_path = os.path.join(settings['wrk_dir'], '{}_ISMN_siteinfo.yml'
                            .format(settings['project_name']))

    if os.path.exists(yml_path):
        os.remove(yml_path)

    with open(yml_path, 'w') as f:
        yaml.dump(ismn_dict, f, default_flow_style=False)


def read_sensor_data(ismn_data, settings, variable=None):
    df = pd.DataFrame()
    for network, station, sensor in ismn_data.collection.iter_sensors(
        variable=variable, depth=settings['swc_depth_range']):
        data = sensor.read_data()
        sensor_df = pd.DataFrame(data)
        sensor_df = sensor_df[sensor_df[f'{variable}_flag'] == 'G'] # filters out bad readings
        sensor_df = sensor_df.resample('D').mean() # resamples to daily values
        sensor_df['DATE'] = sensor_df.index.date
        sensor_df['SITE'] = station.metadata['station'][1]
        df = df.append(sensor_df)

    return df


def initialize(settings_path):

    with open(settings_path) as fp:
        settings = yaml.load(fp, Loader=yaml.FullLoader)

    ismn_data = ISMN_Interface(settings['data_dir'], parallel=True)
    stations = {}
    for n, station in enumerate(ismn_data[settings['network_name']]):
        stations[n] = station.metadata['station'][1]

    # Creates a new dataframe with the station IDs and coordinates
    grid = ismn_data.collection.grid
    gpis, lon, lat = grid.get_grid_points()
    df_coords = pd.DataFrame(index=pd.Index(gpis, name='SITE'), data={'longitude': lon,
                                                                    'latitude': lat})
    df_coords = df_coords.rename(index=stations)

    sm_df = read_sensor_data(ismn_data, settings, variable='soil_moisture')
    sm_df['soil_moisture'] *= 100
    st_df = read_sensor_data(ismn_data, settings, variable='soil_temperature')
    df = pd.merge(sm_df, st_df, how='left', left_on=['DATE', 'SITE'],
                 right_on=['DATE', 'SITE'])

    # Writes snow depth to a dataframe if specified and filters out days with snow cover
    snow_df = pd.DataFrame()
    for network, station, sensor in ismn_data.collection.iter_sensors(
        variable='snow_depth', depth=[0., 0.]):

        data = sensor.read_data()
        sensor_df = pd.DataFrame(data)
        sensor_df.loc[sensor_df['snow_depth']<=0, 'snow'] = 'yes'
        sensor_df = sensor_df.resample('D').count() # counts the entries with snow cover
        sensor_df['DATE'] = sensor_df.index.date
        sensor_df['SITE'] = station.metadata['station'][1]
        sensor_df = sensor_df[['SITE', 'DATE', 'snow']]
        snow_df = snow_df.append(sensor_df)
    try:
        df = pd.merge(df, snow_df, how='left', left_on=['DATE', 'SITE'],
                      right_on=['DATE', 'SITE'])
        df = df[df['snow']==0] # any day with snow cover entries is discarded
    except KeyError:
        pass

    df.rename({'Date_x' : 'DATE'}, axis=1, inplace=True)
    df.set_index('DATE', inplace=True)
    df.rename(columns={'soil_moisture' : 'SWC', 'soil_temperature' : 'TS'}, inplace=True)
    df = df[['SITE', 'SWC', 'TS']]

    ismn_dict = df_coords.T.to_dict()
    for station in stations:
        ismn_dict[stations[station]]['start_date'] = (df[df['SITE'] == stations[station]]
                                                      .index[0].strftime('%Y-%m-%d'))
        ismn_dict[stations[station]]['end_date'] = (df[df['SITE'] == stations[station]]
                                                    .index[-1].strftime('%Y-%m-%d'))
    if settings['do_plot']:
        for station in stations:
            plotting.plot_climate(df[df['SITE'] == stations[station]], settings)

    export(df, ismn_dict, settings)

    print('ISMN preprocessing complete!')
