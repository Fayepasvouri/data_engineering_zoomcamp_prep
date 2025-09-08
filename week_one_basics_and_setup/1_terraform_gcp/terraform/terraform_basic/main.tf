terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "7.1.1"
    }
  }
}

provider "google" {
  credentials = file("creds path .json")
  project     = "zinc-epigram-471514-s3"
  region      = "europe-west2"
}

resource "google_storage_bucket" "data-lake" {
  name          = "zinc-epigram-471514-s3-terra-bucket"
  location      = "EU"
  force_destroy = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30 # days
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_bigquery_dataset" "dataset" {
  dataset_id = "ny_taxi"
  project    = "zinc-epigram-471514-s3"
  location   = "EU"
}