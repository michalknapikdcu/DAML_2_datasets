#!/usr/bin/env python3

import sys
import os
import random
import haversine
import pandas as pd

def prepare_data(dataset_path, length_bottom_threshold, length_top_threshold):

    print('** Reading dataset...')
    db = pd.read_csv(dataset_path)
    print(f'** Done. Read {len(db)} rows.\n')
    print(f'** Printing the header:\n{db.head()}\n')

    # drop columns decided irrelevant for the assignment
    print('** Dropping columns: TRIP_ID CALL_TYPE ORIGIN_CALL ORIGIN_STAND') 
    db.drop(labels=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND'], axis=1, inplace=True)

    # drop missing (reported and other data)
    to_drop = db.MISSING_DATA == True
    print(f'** Dropping {sum(to_drop)} rows with reported missing data.')
    db.drop(db.index[to_drop], inplace=True)
    print(f'** Dropping the MISSING_DATA column.')
    db.drop('MISSING_DATA', axis=1, inplace=True)
    db.dropna(inplace=True)
    print(f'** Dropping NA values, if any: {len(db)} rows left.\n')

    # prepare a series with total route lengths and drop from database the rows with 
    # routes too long or too short
    route_lengths = db.apply(lambda row: len(row.POLYLINE), axis=1)
    route_length_discriminator = ((route_lengths >= length_top_threshold) | \
                                  (route_lengths <= length_bottom_threshold))
    print(f'** Dropping {sum(route_length_discriminator)} rows with '\
          f'routes shorter than {length_bottom_threshold} or longer than '\
          f'{length_top_threshold}.')

    db.drop(db.index[route_length_discriminator], inplace=True)
    print(f'** {len(db)} rows left.')

    # TODO - filter out bad routes based on haversine lengths
    # TODO - normalize GPS data?

if __name__ == '__main__':

    if len(sys.argv) < 6:
        print(f'Usage: {sys.argv[0]} dataset_path datapoints_to_sample '\
              'jump_threshold length_bottom_threshold length_top_threshold')
        sys.exit(1)

    dataset_path = sys.argv[1] # the path to the data file 
    sample_row_count = int(sys.argv[2]) # the number of rows to produce 
    jump_threshold = float(sys.argv[3]) # the maximal valid distance between two adjacent GPS coordinates
    length_bottom_threshold = float(sys.argv[4]) # the minimal valid length of a route 
    length_top_threshold = float(sys.argv[5]) # the maximal valid length of a route 


    print('**')
    print(f'** Collecting {sample_row_count} rows with jump threshold: {jump_threshold}, ')
    print(f'** bottom and top length thresholds: {length_bottom_threshold}, {length_top_threshold}')
    print('**')

    prepare_data(dataset_path, length_bottom_threshold, length_top_threshold)

    print('** all done')
