import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--regionNames', '-n',
                    help='Specify the name of the region to be filtered',
                    required=True,
                    type=str,
                    nargs="+")
parser.add_argument('--regionPopulation', '-p',
                    help='Specify the total population of the area to be filtered',
                    required=True,
                    type=int,
                    nargs="+")
parser.add_argument('--outputFiles', '-o',
                    help='Specify the output CSV file name',
                    required=True,
                    type=str,
                    nargs="+")
args = parser.parse_args()

set_type_list = os.listdir('./data/data_raw/')
for set_type in set_type_list:
    file_list = os.listdir('./data/data_raw/' + set_type)
    i = 0
    for region_name in args.regionNames:
        print("sorting：" + region_name + "start generating" + set_type)
        total_region_file = pd.DataFrame()
        for filename in file_list:
            oneday_file = pd.read_csv('./data/data_raw/' + set_type + '/' + filename)
            oneday_file['city'].fillna(value='not a city', inplace=True)
            oneday_region_file = oneday_file[(oneday_file['province'] == region_name)
                                             & (oneday_file['city'] == 'not a city')]
            total_region_file = pd.concat([total_region_file, oneday_region_file])
        total_region_file.sort_values(by='date', inplace=True)
        # total_region_file['removed'] = total_region_file['cured'] + total_region_file['dead']
        total_region_file.drop(labels=['country', 'countryCode',
                                       'provinceCode', 'city', 'cityCode'],
                               axis=1,
                               inplace=True)
        total_region_file['N'] = args.regionPopulation[i]
        total_region_file['existing infected'] = total_region_file['confirmed'] \
                                           - total_region_file['cured'] \
                                           - total_region_file['dead']
        total_region_file['susceptible'] = total_region_file['N'] \
                                           - total_region_file['existing infected'] \
                                           - total_region_file['cured'] \
                                           - total_region_file['dead']
        order = ['date', 'province', 'N', 'susceptible', 'existing infected', 'cured', 'dead']
        total_region_file = total_region_file[order]
        total_region_file.to_csv(path_or_buf='./data/data_processed/' + set_type + '/'
                                             + args.outputFiles[i],
                                 index=False)
        print('sort Finished, the file has been written to' + args.outputFiles[i])
        i = i + 1
