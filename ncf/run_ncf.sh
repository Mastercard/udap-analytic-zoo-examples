#!/bin/bash
set -x
spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --jars ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.host=localhost \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.memoryOverhead=1024 \
  --conf spark.driver.memoryOverhead=1024 \
  --conf spark.pyspark.python=python3 \
  --conf spark.pyspark.driver.python=python3 \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip,jobs.zip ncf_main.py -n="NCF_DL" -d="/opt/work/data/pcard.csv" -m="/opt/work/model/ncf/" -l="/opt/work/logs/ncf" -s=1 -as="201307" -ae="201401" -vs="201402" -ve="201403" -ts="201403" -te="201404" -is="201405" -ie="201406" -io="/opt/work/output/ncf/" -r=1 -u=10000 -i=200 -o=50 -t=50 -e=50 -b=1600
