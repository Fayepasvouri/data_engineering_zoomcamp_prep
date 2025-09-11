        #!/bin/bash
        gcloud auth activate-service-account --key-file=/Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/zinc-epigram-471514-s3-0827dd59e0b8.json
        bq --project_id zinc-epigram-471514-s3 extract -m de_zoomcamp.tip_model gs://taxi_ml_model_faye/tip_model
        mkdir /tmp/model
        gsutil cp -r gs://taxi_ml_model_faye/tip_model /tmp/model
        mkdir -p serving_dir/tip_model/1
        cp -r /tmp/model/tip_model/* serving_dir/tip_model/1
        docker pull emacski/tensorflow-serving:latest-linux_arm64
        docker run -d -p 8501:8501 \
            --mount type=bind,source=/Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/serving_dir/tip_model,target=/models/tip_model \
            -e MODEL_NAME=tip_model \
            emacski/tensorflow-serving:latest-linux_arm64  
    
        # Weekly at 3am every Monday
        (crontab -l 2>/dev/null; echo "0 3 * * 1 /Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/cron_job_ml_model.sh >> /Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/tip_model_cron.log 2>&1") | crontab -
        # Monthly at 3am on the 1st
        (crontab -l 2>/dev/null; echo "0 3 1 * * /Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/cron_job_ml_model.sh >> /Users/faye.pasvouri/data_engineering_zoomcamp_prep/week_three_data_warehouse/tip_model_cron.log 2>&1") | crontab -
        echo "Cron jobs scheduled successfully."
