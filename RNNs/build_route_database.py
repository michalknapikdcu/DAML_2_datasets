#!/usr/bin/env python3

import sys
import os
import random
import haversine
import pandas as pd

def prepare_data(dataset_path, length_bottom_threshold, length_top_threshold, \
                 sample_row_count, minimal_ride_req, jump_threshold):

    # read and summarize data
    print('** Reading dataset...')
    db = pd.read_csv(dataset_path)
    print(f'** Done. Read {len(db)} rows.\n')
    print(f'** Printing columns with their datatypes:\n{db.dtypes}\n')
    print(f'** Printing the header:\n{db.head()}\n')

    # drop columns decided irrelevant for the assignment
    print('** Dropping columns: TRIP_ID CALL_TYPE ORIGIN_CALL ORIGIN_STAND DAY_TYPE') 
    db.drop(labels=['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'DAY_TYPE'], \
            axis=1, inplace=True)

    # drop missing (reported and other data)
    to_drop = db.MISSING_DATA == True
    print(f'** Dropping {sum(to_drop)} rows with reported missing data.')
    db.drop(db.index[to_drop], inplace=True)
    print(f'** Dropping the MISSING_DATA column.')
    db.drop('MISSING_DATA', axis=1, inplace=True)
    db.dropna(inplace=True)
    print(f'** Dropping NA values, if any.\n{len(db)} rows left.\n')

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
    db = db.groupby('TAXI_ID').sample(n=group_sample_count)
    print(f'** {len(db)} rows left.\n')

    # compute distances between adjacent steps in routes using haversine formula
    def get_distances(polyline):
        # polyline data entry in the dataset is a string of form '[[1.2, 3.4], [4.5, 6.89],]'
        # and needs to be evaluated; this is costly, so done only after sampling
        polyline = eval(polyline)
        
        return ([haversine.haversine(p, n, unit=haversine.Unit.METERS) \
                for p,n in zip(polyline[1:], polyline[:-1])])
    
    # remove anomalous readings
    print(f'** Removing rows with adjacent position readings further than {jump_threshold}.')
    # get selector for the rows where an anomalous reading is detected
    def are_distances_anomalous(polyline):
        adj_distances = get_distances(polyline)
        return any(distance > jump_threshold for distance in adj_distances)

    anomaly_discriminator = db.apply(lambda row: are_distances_anomalous(row.POLYLINE), axis=1)
    anomalous_reading_count = sum(anomaly_discriminator)
    print(f'** There are {anomalous_reading_count} such entries.')
    db.drop(db.index[anomaly_discriminator], inplace=True)
    print(f'** {len(db)} rows left.\n')

    # max-min center the route data 
    # get minimal and maximal values of GPS coordinates per row
    print(f'** Applying min-max feature scaling to GPS coordinates.')
    def get_boundary_gps_vals(polyline):
        polyline = eval(polyline)
        latitudes = [entry[0] for entry in polyline] 
        longitudes = [entry[1] for entry in polyline] 
        return min(latitudes), max(latitudes), min(longitudes), max(longitudes)

    gps_maxima_per_row = db.apply(lambda row: get_boundary_gps_vals(row.POLYLINE), axis=1)
    min_lat = gps_maxima_per_row.apply(lambda row: row[0]).min()
    max_lat = gps_maxima_per_row.apply(lambda row: row[1]).max()
    min_long = gps_maxima_per_row.apply(lambda row: row[2]).min()
    max_long = gps_maxima_per_row.apply(lambda row: row[3]).max()
    lat_span = max_lat - min_lat
    long_span = max_long - min_long

    def center_route(polyline):
        polyline = eval(polyline)
        centered_polyline = [[(entry[0] - min_lat)/lat_span, (entry[1] - min_long)/long_span] for entry in polyline]
        return centered_polyline
        
    centered_entries = db.apply(lambda row: center_route(row.POLYLINE), axis=1)
    db = db.assign(POLYLINE = centered_entries)
    db.reset_index(drop=True, inplace=True)
    print(f'** Printing the header of the transformed dataset:\n{db.head()}\n')

    return db
 
if __name__ == '__main__':

    if len(sys.argv) < 8:
        print(f'Usage: {sys.argv[0]} dataset_path datapoints_to_sample '\
              'jump_threshold length_bottom_threshold length_top_threshold '\
              'minimal_ride_req target_file_name')
        sys.exit(1)

    dataset_path = sys.argv[1] # the path to the data file 
    sample_row_count = int(sys.argv[2]) # the approximate number of rows to produce 
    jump_threshold = float(sys.argv[3]) # the maximal valid distance (in meters) between two adjacent GPS  
    length_bottom_threshold = float(sys.argv[4]) # the minimal valid number of steps in a route 
    length_top_threshold = float(sys.argv[5]) # the maximal valid number of steps in a route 
    minimal_ride_req = int(sys.argv[6]) # the minimal number of rides required from a taxi
    target_file_name = sys.argv[7] # the name of csv file to save prepared data

    print('**')
    print(f'** Sampling {sample_row_count} rows with jump threshold: {jump_threshold}, ')
    print(f'** bottom and top length thresholds: {length_bottom_threshold}, {length_top_threshold}')
    print(f'** and at least {minimal_ride_req} rides required from a taxi.')
    print('**\n')

    transformed_data = prepare_data(dataset_path, length_bottom_threshold, length_top_threshold, \
                                    sample_row_count, minimal_ride_req, jump_threshold)

    transformed_data.to_csv(target_file_name)

    print(f'** Written the transformed dataset to {target_file_name}. All done.')

