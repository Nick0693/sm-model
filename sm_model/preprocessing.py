import os

import yaml
import pandas as pd

import ee_parser

"""
Preprocessing workflow for merging ICOS and/or ISMN soil moisture point data
with remotely sensed indices through the Google Earth Engine.

The output is a dataframe (csv) that can be ingested into the deep learning model.

All configurable settings can be found in the associated settings.yml file
"""

def compile_data(settings_path):

    with open(settings_path) as fp:
        settings = yaml.load(fp, Loader=yaml.FullLoader)

    if settings['data_source'] == 'ICOS':
        df_sm = pd.read_csv(os.path.join(
            settings['wrk_dir'], '{}_ICOS_Dataframe.csv'.format(settings['project_name'])), index_col='DATE')
        with open(os.path.join(settings['wrk_dir'], '{}_ICOS_siteinfo.yml'.format(settings['project_name']))) as f:
            site_info = yaml.load(f, Loader=yaml.FullLoader)

    elif settings['data_source'] == 'ISMN':
        df_sm = pd.read_csv(os.path.join(
            settings['wrk_dir'], '{}_ISMN_Dataframe.csv'.format(settings['project_name'])), index_col='DATE')
        with open(os.path.join(settings['wrk_dir'], '{}_ISMN_siteinfo.yml'.format(settings['project_name']))) as f:
            site_info = yaml.load(f, Loader=yaml.FullLoader)

    variable_list = settings['variables']
    if ~settings['overwrite_variables']:
        for var in settings['variables']:
            if var in df_sm.columns and var in variable_list:
                variable_list.remove(var)

    df = pd.DataFrame()
    print('Extracting data from Earth Engine...')
    i_start, i_end = 1, len(site_info)
    for station in site_info:
        print(f'  Station {station} ({i_start} of {i_end})')
        df_sm_subset = df_sm[df_sm['SITE'] == station]
        df_sm_subset.index = pd.to_datetime(df_sm_subset.index)
        try:
            df_ee = (ee_parser.assemble_variables(
                site_info[station]['longitude'], site_info[station]['latitude'],
                site_info[station]['start_date'], site_info[station]['end_date'],
                variable_list, station, settings))
            df_ee = df_ee.reindex(pd.date_range(
                start=site_info[station]['start_date'], end=site_info[station]['end_date'], freq='D'))
            df_subset = pd.merge(df_ee, df_sm_subset, how='left', left_index=True, right_index=True)
            df_subset.rename({'SITE_x' : 'SITE'}, axis=1, inplace=True)
            df_subset.drop('SITE_y', axis=1, inplace=True)
            df = pd.concat([df, df_subset])
        except (KeyError, AttributeError):
            pass

        i_start += 1

    df.index.name = 'Date'

    base_variables = ['SITE', 'SWC', 'TS', 'TA', 'P']
    for var in settings['variables']:
        if var not in base_variables:
            base_variables.append(var)

    df = df[base_variables]
    df.to_csv(os.path.join(settings['wrk_dir'], '{}_compiled_dataframe.csv'
                           .format(settings['project_name'])))
