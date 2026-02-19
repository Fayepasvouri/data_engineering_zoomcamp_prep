# gsutil cp adverity_analytics_job.py gs://my-dataproc-bucket/jobs/

# gcloud dataproc clusters create adverity-analytics \
  --region=us-central1 \
  --zone=us-central1-a \
  --single-node \
  --image-version=2.1-debian12 \
  --enable-component-gateway \
  --properties=spark:spark.sql.adaptive.enabled=true

# gcloud dataproc jobs submit pyspark \
  gs://my-dataproc-bucket/jobs/adverity_analytics_job.py \
  --cluster=adverity-analytics \
  --region=us-central1

# Schedule via Cloud Composer (Airflow)

# Run after dbt finishes

# Add:

- partition overwrite (by date)

- incremental logic

- data quality checks

- > 1 billion	Start considering Spark

- 5B+	Spark usually cheaper