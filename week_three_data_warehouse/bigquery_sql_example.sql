-- Check yellow trip data
SELECT * FROM public.yellow_tripdata limit 10;

-- Create a non partitioned table from external table
CREATE TABLE public.yellow_tripdata_non_partitioned AS
SELECT * FROM public.yellow_tripdata;

-- Create a partitioned table from external table
CREATE TABLE public.yellow_tripdata_partitioned (
    LIKE public.yellow_tripdata INCLUDING ALL
) PARTITION BY RANGE (tpep_pickup_datetime);

-- Scanning 1.6GB of data
SELECT DISTINCT(VendorID)
FROM public.yellow_tripdata_non_partitioned
WHERE DATE(tpep_pickup_datetime) BETWEEN '2019-06-01' AND '2019-06-30';

-- Scanning ~106 MB of DATA
SELECT DISTINCT(VendorID)
FROM public.yellow_tripdata_partitioned
WHERE DATE(tpep_pickup_datetime) BETWEEN '2019-06-01' AND '2019-06-30';


-- Query scans 1.1 GB
SELECT count(*) as trips
FROM public.yellow_tripdata_partitioned
WHERE DATE(tpep_pickup_datetime) BETWEEN '2019-06-01' AND '2020-12-31'
  AND VendorID=1;

-- Query scans 864.5 MB
SELECT count(*) as trips
FROM public.yellow_tripdata_partitioned_clustered
WHERE DATE(tpep_pickup_datetime) BETWEEN '2019-06-01' AND '2020-12-31'
  AND VendorID=1;

CREATE TABLE public.yellow_tripdata_partitioned (
    LIKE public.yellow_tripdata INCLUDING ALL
) PARTITION BY RANGE (tpep_pickup_datetime);

Step 2: Create partitions (example: Jan and Feb 2019)
CREATE TABLE public.yellow_tripdata_2019_01
    PARTITION OF public.yellow_tripdata_partitioned
    FOR VALUES FROM ('2019-01-01') TO ('2019-02-01');

CREATE TABLE public.yellow_tripdata_2019_02
    PARTITION OF public.yellow_tripdata_partitioned
    FOR VALUES FROM ('2019-02-01') TO ('2019-03-01');

Step 4: If you want clustering (optional)
CREATE INDEX idx_yellow_tripdata_vendorid_2019_01 
    ON public.yellow_tripdata_2019_01 (VendorID);

CLUSTER public.yellow_tripdata_2019_01 
    USING idx_yellow_tripdata_vendorid_2019_01;
