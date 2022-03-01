import os
import glob
import fnmatch

import yaml
import numpy as np
import pandas as pd

import plotting

"""
Reads and aggregates soil moisture, soil temperature, air temperature, and
precipitation for multiple ICOS sensors

This is not a downloader so data should be downloaded beforehand from here:
https://www.icos-cp.eu/data-services/about-data-portal

All configurable data can be found in the associated settings.yml file.
"""

def export(df, icos_dict, settings):
    df.to_csv(os.path.join(settings['wrk_dir'], '{}_ICOS_Dataframe.csv'
                           .format(settings['project_name'])), index=True)

    yml_path = os.path.join(settings['wrk_dir'], '{}_ICOS_siteinfo.yml'
                            .format(settings['project_name']))

    if os.path.exists(yml_path):
        os.remove(yml_path)

    with open(yml_path, 'w') as f:
        yaml.dump(icos_dict, f, default_flow_style=False)


def initialize(settings_path):

    with open(settings_path) as fp:
        settings = yaml.load(fp, Loader=yaml.FullLoader)

    master_df = pd.DataFrame()
    icos_dict = {}
    icos_dir = settings['data_dir']
    meteo_csv_list = list(set(glob.glob(f'{icos_dir}\**\*_METEO_01.csv')) -
                          set(glob.glob(f'{icos_dir}\**\*VARINFO_METEO_01.csv')))

    for meteo in meteo_csv_list:
        # Reading meteo CSV and formatting date/index/NaNs
        df = pd.read_csv(meteo)

        try:
            columns = list(df.columns)
            swc = fnmatch.filter(columns, 'SWC_?')
            ts = fnmatch.filter(columns, 'TS_?')
            ta = fnmatch.filter(columns, 'TA') + fnmatch.filter(columns, 'TA_?')
            use_columns = ['TIMESTAMP_START', 'P'] + ta + swc + ts
            df = df[use_columns]
            df['DATE'] =  pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
            df.set_index(df['DATE'], inplace=True)
            df.drop(columns=['TIMESTAMP_START', 'DATE'], inplace=True)
            df.replace(to_replace=-9999, value=np.nan, inplace=True)

            # Resampling for daily values and averaging SWC/TS across all probes
            df_P = df['P'].resample('D').sum()
            df = df.resample('D').mean()
            df['P'] = df_P
            df['SWC'] = df[swc].mean(axis=1)
            df['TS'] = df[ts].mean(axis=1)
            df['TA'] = df[ta].mean(axis=1)

            try:
                ta.remove('TA')

            except ValueError:
                pass

            df.drop(columns=swc+ts+ta, inplace=True)
            df[df['TS'] >= 0] # removes days with frozen soil

            # Retrieving WGS84 coordinates from the associated site info file
            id_ = meteo.split('\\')[-1].split('_')[1]
            site_csv = glob.glob(f'{icos_dir}\**\*{id_}_SITEINFO.csv')[0]
            site_info = pd.read_csv(site_csv, index_col='VARIABLE')
            lat = float(site_info.loc['LOCATION_LAT'].DATAVALUE)
            lon = float(site_info.loc['LOCATION_LONG'].DATAVALUE)
            site_name = site_info.loc['SITE_NAME'].DATAVALUE

            df['SITE'] = site_name
            icos_dict[site_name] = {
                'latitude' : lat,
                'longitude' : lon,
                'start_date' : df.index[0].strftime('%Y-%m-%d'),
                'end_date' : df.index[-1].strftime('%Y-%m-%d')
            }

            df = df[['SITE', 'SWC', 'TS', 'TA', 'P']]

            if settings['do_plot']:
                plotting.plot_climate(df, settings)

            master_df = pd.concat([master_df, df])

        except KeyError as e:
            print(e, ' for ', meteo.split('\\')[-1].split('_')[1])

    export(master_df, icos_dict, settings)

    print('ICOS preprocessing complete!')
