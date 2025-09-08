variable "credentials" {
  description = "Path to the GCP credentials JSON file"
  type        = string
  default     = "creds path"
}

variable "project" {
  description = "GCP project ID"
  type        = string
  default     = "zinc-epigram-471514-s3"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west2"
}

variable "gcs_bucket_name" {
  description = "GCS bucket name"
  type        = string
  default     = "zinc-epigram-471514-s3-terra-bucket"
}

variable "bq_dataset_name" {
  description = "BigQuery dataset name"
  type        = string
  default     = "my_data"
}

variable "location" {
  description = "GCP region"
  type        = string
  default     = "EU"
}

variable "gcs_storage_class" {
  description = "Bucket Storage Class"
  default     = "STANDARD"
}