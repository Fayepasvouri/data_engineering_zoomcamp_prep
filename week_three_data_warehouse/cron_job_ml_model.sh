# --- NEW: Cron job scheduling ---

- Steps
    - name: Create a shell script for the workflow
      run: |
        cat <<'EOF' > ~/run_tip_model.sh
        #!/bin/bash
        gcloud auth activate-service-account --key-file=week_three_data_warehouse/zinc-epigram-471514-s3-0827dd59e0b8.json
        bq --project_id zinc-epigram-471514-s3 extract -m de_zoomcamp.tip_model gs://taxi_ml_model/tip_model
        mkdir -p /tmp/model
        gsutil cp -r gs://taxi_ml_model/tip_model /tmp/model
        mkdir -p serving_dir/tip_model/1
        cp -r /tmp/model/tip_model/* serving_dir/tip_model/1
        docker stop tip_model || true
        docker run -d -p 8501:8501 --mount type=bind,source=$(pwd)/serving_dir/tip_model,target=/models/tip_model -e MODEL_NAME=tip_model tensorflow/serving
        EOF
        chmod +x ~/run_tip_model.sh

    - name: Schedule cron job
      run: |
        # Weekly at 3am every Monday
        (crontab -l 2>/dev/null; echo "0 3 * * 1 ~/run_tip_model.sh >> ~/tip_model_cron.log 2>&1") | crontab -
        # Monthly at 3am on the 1st
        (crontab -l 2>/dev/null; echo "0 3 1 * * ~/run_tip_model.sh >> ~/tip_model_cron.log 2>&1") | crontab -