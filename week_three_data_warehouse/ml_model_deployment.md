# ML Model Deployment in GCP Cloud 

- Steps
    - gcloud auth login
    - bq --project_id zinc-epigram-471514-s3 extract -m de_zoomcamp.tip_model gs://taxi_ml_model_faye/tip_model
    - mkdir /tmp/model
    - gsutil cp -r gs://taxi_ml_model_faye/tip_model /tmp/model
    - mkdir -p serving_dir/tip_model/1
    - cp -r /tmp/model/tip_model/* serving_dir/tip_model/1
    - docker pull emacski/tensorflow-serving:latest-linux_arm64

    - docker run -d -p 8501:8501 \
        --mount type=bind,source=/Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/serving_dir/tip_model,target=/models/tip_model \
        -e MODEL_NAME=tip_model \
        emacski/tensorflow-serving:latest-linux_arm64
        
    - curl -d '{"instances": [{"passenger_count":1, "trip_distance":12.2, "PULocationID":"193", "DOLocationID":"264", "payment_type":"2","fare_amount":20.4,"tolls_amount":0.0}]}' -X POST http://localhost:8501/v1/models/tip_model:predict
    - http://localhost:8501/v1/models/tip_model