  SELECT
    passenger_count,
    trip_distance,
    PULocationID,
    DOLocationID,
    payment_type,
    fare_amount,
    tolls_amount,
    tip_amount
  FROM
    `zinc-epigram-471514-s3.de_zoomcamp.yellow_tripdata_2019_01`
  WHERE
    fare_amount != 0;
  CREATE OR REPLACE TABLE
    `de_zoomcamp.yellow_tripdata_ml` ( `passenger_count` INT64,
      `trip_distance` FLOAT64,
      `PULocationID` STRING,
      `DOLocationID` STRING,
      `payment_type` STRING,
      `fare_amount` FLOAT64,
      `tolls_amount` FLOAT64,
      `tip_amount` FLOAT64 ) AS (
    SELECT
      passenger_count,
      trip_distance,
      CAST(PULocationID AS STRING),
      CAST(DOLocationID AS STRING),
      CAST(payment_type AS STRING),
      fare_amount,
      tolls_amount,
      tip_amount
    FROM
      `de_zoomcamp.yellow_tripdata_2019_01`
    WHERE
      fare_amount != 0 );

      
-- CREATE MODEL WITH DEFAULT SETTING
  CREATE OR REPLACE MODEL
    `de_zoomcamp.tip_model` OPTIONS (
      model_type='linear_reg',
      input_label_cols=['tip_amount'],
      DATA_SPLIT_METHOD='AUTO_SPLIT') AS
  SELECT
    *
  FROM
    `de_zoomcamp.yellow_tripdata_ml`
  WHERE
    tip_amount IS NOT NULL;
  SELECT * FROM ML.FEATURE_INFO(MODEL `de_zoomcamp.tip_model`);

-- EVALUATE THE MODEL
SELECT
  *
FROM
  ML.EVALUATE(MODEL `de_zoomcamp.tip_model`,
    (
    SELECT
      *
    FROM
      `de_zoomcamp.yellow_tripdata_ml`
    WHERE
      tip_amount IS NOT NULL ));

-- PREDICT THE MODEL
SELECT
  *
FROM
  ML.PREDICT(MODEL `de_zoomcamp.tip_model`,
    (
    SELECT
      *
    FROM
      `de_zoomcamp.yellow_tripdata_ml`
    WHERE
      tip_amount IS NOT NULL ));

-- PREDICT AND EXPLAIN
SELECT
  *
FROM
  ML.EXPLAIN_PREDICT(MODEL `de_zoomcamp.tip_model`,
    (
    SELECT
      *
    FROM
      `de_zoomcamp.yellow_tripdata_ml`
    WHERE
      tip_amount IS NOT NULL ),
    STRUCT(3 AS top_k_features));
  -- HYPER PARAM TUNNING
CREATE OR REPLACE MODEL
  `de_zoomcamp.tip_hyperparam_model` OPTIONS (model_type='linear_reg',
    input_label_cols=['tip_amount'],
    DATA_SPLIT_METHOD='AUTO_SPLIT',
    num_trials=5,
    max_parallel_trials=2,
    l1_reg=hparam_range(0,
      20),
    l2_reg=hparam_candidates([0,
      0.1,
      1,
      10])) AS
SELECT
  *
FROM
  `de_zoomcamp.yellow_tripdata_ml`
WHERE
  tip_amount IS NOT NULL;