#!/bin/bash
set -x
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo \
  --master local[4] \
  --driver-memory 8g \
  --conf spark.executor.memoryOverhead=1024 \
  --conf spark.driver.memoryOverhead=1024 \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip,jobs.zip \
   ncf_tf2_main.py -n="NCF_DL" -d="/opt/work/data/pcard.csv" -m="/opt/work/model/ncf/" \
   -l="/opt/work/logs/ncf" -s=1 -as="201307" -ae="201401" -vs="201402" -ve="201403" \
   -ts="201403" -te="201404" -is="201405" -ie="201406" -io="/opt/work/output/ncf/" \
   -r=1 -u=10000 -i=200 -o=50 -t=50 -e=50 -b=1600
