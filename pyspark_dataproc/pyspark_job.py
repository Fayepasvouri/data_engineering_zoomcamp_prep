from pyspark.sql import SparkSession, functions as F, Window
import re

# ===============================
# 1. Spark session
# ===============================
spark = (
    SparkSession.builder
    .appName("Adverity Unified Performance Analytics")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

# ===============================
# 2. Client inference from GCP project
# ===============================
PROJECT_ID = spark.conf.get("spark.hadoop.fs.gs.project.id")

def get_client(project_id: str) -> str:
    match = re.search(r'client[-_]?([a-z0-9]+)', project_id.lower())
    return match.group(1) if match else "unknown_client"

CLIENT = get_client(PROJECT_ID)

BQ_DBT = f"{PROJECT_ID}.dbt_marts"
BQ_OUT = f"{PROJECT_ID}.analytics"

# ===============================
# 3. Load dbt standardised fact table
# ===============================
fact_ads = (
    spark.read.format("bigquery")
    .option("table", f"{BQ_DBT}.fact_ads_performance")
    .load()
    .withColumn("client", F.lit(CLIENT))
)

# ===============================
# 4. Core KPI enrichment
# ===============================
ads = (
    fact_ads
    .withColumn("ctr", F.col("clicks") / F.col("impressions"))
    .withColumn("cpc", F.col("spend_converted") / F.col("clicks"))
    .withColumn("cpm", F.col("spend_converted") * 1000 / F.col("impressions"))
    .withColumn("cpa", F.col("spend_converted") / F.col("conversions"))
    .withColumn("roas", F.col("revenue_converted") / F.col("spend_converted"))
)

# Cache once – reused many times
ads.cache()

# ===============================
# 5. Unified aggregation layer
# ===============================

aggregations = {}

# A. Daily performance
aggregations["daily_performance"] = (
    ads.groupBy("client", "date")
    .agg(
        F.sum("spend_converted").alias("spend"),
        F.sum("revenue_converted").alias("revenue"),
        F.sum("impressions").alias("impressions"),
        F.sum("clicks").alias("clicks"),
        F.sum("conversions").alias("conversions"),
        F.sum("video_views").alias("video_views"),
        F.sum("engagements").alias("engagements"),
    )
    .withColumn("ctr", F.col("clicks") / F.col("impressions"))
    .withColumn("roas", F.col("revenue") / F.col("spend"))
)

# B. Platform / channel
aggregations["channel_performance"] = (
    ads.groupBy("client", "platform", "channel")
    .agg(
        F.sum("spend_converted").alias("spend"),
        F.sum("revenue_converted").alias("revenue"),
        F.sum("conversions").alias("conversions"),
        F.sum("clicks").alias("clicks"),
    )
    .withColumn("roas", F.col("revenue") / F.col("spend"))
    .withColumn("cpa", F.col("spend") / F.col("conversions"))
)

# C. Account level
aggregations["account_performance"] = (
    ads.groupBy(
        "client",
        "platform",
        "account_id",
        "account_name"
    )
    .agg(
        F.sum("spend_converted").alias("spend"),
        F.sum("revenue_converted").alias("revenue"),
        F.sum("clicks").alias("clicks"),
        F.sum("conversions").alias("conversions"),
    )
    .withColumn("roas", F.col("revenue") / F.col("spend"))
)

# D. Campaign funnel
aggregations["campaign_funnel"] = (
    ads.groupBy(
        "client",
        "platform",
        "campaign_id",
        "campaign_name",
        "campaign_objective"
    )
    .agg(
        F.sum("impressions").alias("impressions"),
        F.sum("clicks").alias("clicks"),
        F.sum("conversions").alias("conversions"),
        F.sum("spend_converted").alias("spend"),
        F.sum("revenue_converted").alias("revenue"),
    )
    .withColumn("ctr", F.col("clicks") / F.col("impressions"))
    .withColumn("conversion_rate", F.col("conversions") / F.col("clicks"))
    .withColumn("roas", F.col("revenue") / F.col("spend"))
)

# E. Geo x device
aggregations["geo_device_performance"] = (
    ads.groupBy("client", "geo", "device")
    .agg(
        F.sum("spend_converted").alias("spend"),
        F.sum("revenue_converted").alias("revenue"),
        F.sum("conversions").alias("conversions"),
    )
    .withColumn("roas", F.col("revenue") / F.col("spend"))
)

# F. Trends (WoW)
w = Window.partitionBy("client").orderBy("date")

aggregations["performance_trends"] = (
    aggregations["daily_performance"]
    .withColumn("prev_spend", F.lag("spend").over(w))
    .withColumn("wow_spend_change",
                (F.col("spend") - F.col("prev_spend")) / F.col("prev_spend"))
)

# ===============================
# 6. Write all outputs in one loop
# ===============================
for table, df in aggregations.items():
    (
        df.write
        .format("bigquery")
        .option("table", f"{BQ_OUT}.{table}")
        .mode("overwrite")
        .save()
    )

spark.stop()
