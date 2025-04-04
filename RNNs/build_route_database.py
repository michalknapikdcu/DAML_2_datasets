#!/usr/bin/env python3

import sys
import os
import random
import haversine
import pandas as pd

def prepare_data(dataset_path, length_bottom_threshold, length_top_threshold, \
                 sample_row_count, minimal_ride_req):

    # read and summarize data
    print('** Reading dataset...')
    db = pd.read_csv(dataset_path)
    print(f'** Done. Read {len(db)} rows.\n')
    print(f'** Printing columns with their datatypes:\n{db.dtypes}\n')
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
    # routes too long or too short; 
    # there is a trick here: these entries are Pandas objects, i.e. strings; so to find
    # the length of the sequence of coordinates like '[[1.2, 2.3], [1.23, 4.55], ...]'
    # it suffices to count '[' and subtract 1
    route_lengths = db.apply(lambda row: row.POLYLINE.count('[') - 1, axis=1)
    route_length_discriminator = ((route_lengths >= length_top_threshold) | \
                                  (route_lengths <= length_bottom_threshold))
    print(f'** Dropping {sum(route_length_discriminator)} rows with '\
          f'routes shorter than {length_bottom_threshold} or longer than '\
          f'{length_top_threshold}.')
    db.drop(db.index[route_length_discriminator], inplace=True)
    print(f'** {len(db)} rows left.\n')

    # drop rows with taxi drivers whose activity is below the given threshold
    taxi_id_count = len(db.TAXI_ID.unique())
    rides_count = db.groupby('TAXI_ID').size()
    active_taxi_ids = rides_count[rides_count >= minimal_ride_req]
    active_taxi_count = len(active_taxi_ids)
    inactive_driver_count = taxi_id_count - active_taxi_count
    inactive_taxi_discriminator = db.apply(lambda row: row['TAXI_ID'] not in active_taxi_ids, axis=1) 
    print(f'** Dropping all rows with taxi ids who have less than {minimal_ride_req} rides.')
    print(f'** There are {inactive_driver_count} of {taxi_id_count} such taxis.')
    db.drop(db.index[inactive_taxi_discriminator], inplace=True)
    print(f'** {len(db)} rows left.\n')

    # now, sample the same number of rows for each taxi id, to get approximately 
    # the requested number of rows in total, evenly distributed over ids
    group_sample_count = sample_row_count//active_taxi_count
    print(f'** Sampling {group_sample_count} from each taxi id, '\
          f'to get approximately {sample_row_count} rows.')
    db = db.groupby('TAXI_ID').sample(n=group_sample_count).count()
    print(f'** {len(db)} rows left.\n')

    return
    # compute distances between adjacent steps in routes using haversine formula
    def get_distances(polyline):
        print(list(zip(polyline[1:], polyline[:-1])))
        print([haversine.haversine(p, n, unit=haversine.Unit.METERS) \
                for p,n in zip(polyline[1:], polyline[:-1])])

    adj_distances = db.apply(lambda row: get_distances(row.POLYLINE), axis=1)
    print(adj_distances)
   
    # TODO - filter out bad routes based on haversine lengths
    # TODO - normalize GPS data?

if __name__ == '__main__':

    if len(sys.argv) < 7:
        print(f'Usage: {sys.argv[0]} dataset_path datapoints_to_sample '\
              'jump_threshold length_bottom_threshold length_top_threshold '\
              'minimal_ride_req')
        sys.exit(1)

    dataset_path = sys.argv[1] # the path to the data file 
    sample_row_count = int(sys.argv[2]) # the approximate number of rows to produce 
    jump_threshold = float(sys.argv[3]) # the maximal valid distance between two adjacent GPS coordinates
    length_bottom_threshold = float(sys.argv[4]) # the minimal valid number of steps in a route 
    length_top_threshold = float(sys.argv[5]) # the maximal valid number of steps in a route 
    minimal_ride_req = int(sys.argv[6]) # the minimal number of rides required from a taxi

    print('**')
    print(f'** Collecting {sample_row_count} rows with jump threshold: {jump_threshold}, ')
    print(f'** bottom and top length thresholds: {length_bottom_threshold}, {length_top_threshold}')
    print(f'** and at least {minimal_ride_req} rides required from a taxi.')
    print('**\n')

    prepare_data(dataset_path, length_bottom_threshold, length_top_threshold, \
                 sample_row_count, minimal_ride_req)

    print('** all done')
