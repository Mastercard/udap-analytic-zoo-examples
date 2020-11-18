import random
import time
import datetime
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, col, udf, lit
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import FloatType,DoubleType,ArrayType
from pyspark.ml.feature import VectorAssembler

__author__ = 'e047349'

def _date_to_month(date):
    return (int(date) // 100 - 2000) * 12 + int(date) % 100


def _generate_id_pair(u_limit,m_limit):
    uid = random.randint(1, u_limit)
    mid = random.randint(1, m_limit)
    return (uid, mid)
	
def load_table(spark,query,u_limit,m_limit):
    sql_df = spark.sql(query)
    data_df = sql_df.na.drop()
    userIndexer = StringIndexer(inputCol="u", outputCol="uid").fit(data_df)
    itemIndexer = StringIndexer(inputCol="m", outputCol="mid").fit(data_df)
    data_df = itemIndexer.transform(userIndexer.transform(data_df)) \
	    .withColumn("uid", (col("uid") + 1).cast(FloatType())) \
	    .withColumn("mid", (col("mid") + 1).cast(FloatType())) \
	    .cache()

    month_seq_udf = udf(lambda s: _date_to_month(s))
    uDF = data_df.select("uid", "u").distinct().orderBy("uid")
    mDF = data_df.select("mid", "m").distinct().orderBy("mid")
    tDF = data_df.filter(data_df["uid"] <= u_limit).filter(data_df["mid"] <= m_limit) \
        .withColumn("month", col("transaction_month")) \
        .drop("u", "m")
    return uDF, mDF, tDF

def load_csv(spark,data_source_path,u_limit,m_limit):
    raw_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(data_source_path)

    data_df = raw_df.select("Cardholder Last Name",
                            "Cardholder First Initial",
                            "Amount",
                            "Vendor",
                            "Year-Month") \
        .select(
        concat(col("Cardholder Last Name"), lit(" "), col("Cardholder First Initial")).alias("u"),
        concat(col("Vendor")).alias("m"),
        col("Year-Month").alias("date"),
        col("Amount")
    )

    userIndexer = StringIndexer(inputCol="u", outputCol="uid").fit(data_df)
    itemIndexer = StringIndexer(inputCol="m", outputCol="mid").fit(data_df)

    data_df = itemIndexer.transform(userIndexer.transform(data_df)) \
        .withColumn("uid", (col("uid") + 1).cast(FloatType())) \
        .withColumn("mid", (col("mid") + 1).cast(FloatType())) \
        .cache()

    month_seq_udf = udf(lambda s: _date_to_month(s))
    uDF = data_df.select("uid", "u").distinct().orderBy("uid")
    mDF = data_df.select("mid", "m").distinct().orderBy("mid")
    tDF = data_df.filter(data_df["uid"] <= u_limit).filter(data_df["mid"] <= m_limit) \
        .withColumn("month", month_seq_udf(col("date"))) \
        .drop("u", "m")
    return uDF, mDF, tDF


def genData(tDF,sc,spark,label_start_date, label_end_date,neg_rate,sliding_length,u_limit,m_limit):
    label_start_month = _date_to_month(label_start_date)
    label_end_month = _date_to_month(label_end_date)
    transaction_start_month = label_start_month - sliding_length
    tdf_in_feature_range = tDF.filter(tDF["month"] < label_start_month) \
                      .filter(tDF["month"] >= transaction_start_month)

    umFreq_df = tdf_in_feature_range.groupBy("uid", "mid").count()
    slidingDFs = []
    for label_month in range(label_start_month, label_end_month):
        positiveID_df = tDF.filter(tDF["month"] == label_month) \
            .select("uid", "mid").distinct()
        posCount = positiveID_df.count()

        list = sc.parallelize(range(1, posCount * neg_rate)).map(lambda x: _generate_id_pair(u_limit,m_limit))
        negativeID_df = spark.createDataFrame(list).select(col("_1").alias("uid"), col("_2").alias("mid")) \
            .distinct().subtract(positiveID_df)
        label_df = positiveID_df.withColumn("labels", lit(0.0)).union(
            negativeID_df.withColumn("labels", lit(1.0))
        )  # tensorflow use 0 based label
        #print("pos: ", posCount, "neg: ", negativeID_df.count())
        #print("label df count: ", label_df.count())
        featureDF = label_df.join(umFreq_df, ["uid", "mid"], how="left").na.fill(0)
        slidingDFs.append(featureDF)

    resultDF = slidingDFs[0]
    for i in range(1, len(slidingDFs)):
        resultDF = resultDF.union(slidingDFs[i])


    assemble_udf = udf(lambda x, y, z: [float(x), float(y), float(z)], ArrayType(DoubleType(), False))
    resultDF = resultDF.withColumn("features", assemble_udf(col("uid"), col("mid"), col("count")))

    return resultDF
