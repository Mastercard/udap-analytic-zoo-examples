#!/bin/bash

set -x

#setup pathes
ANALYTICS_ZOO_TUTORIALS_HOME=/opt/work/ncf
SPARK_MAJOR_VERSION=${SPARK_VERSION%%.[0-9]}
echo $ANALYTICS_ZOO_TUTORIALS_HOME
echo $ANALYTICS_ZOO_VERSION
echo $BIGDL_VERSION
echo $SPARK_VERSION
echo $SPARK_MAJOR_VERSION

export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=$ANALYTICS_ZOO_TUTORIALS_HOME --ip=0.0.0.0 --port=$NotebookPort --no-browser --NotebookApp.token=$NotebookToken --allow-root"

echo $RUNTIME_SPARK_MASTER
echo $RUNTIME_EXECUTOR_CORES
echo $RUNTIME_DRIVER_CORES
echo $RUNTIME_DRIVER_MEMORY
echo $RUNTIME_EXECUTOR_CORES
echo $RUNTIME_EXECUTOR_MEMORY
echo $RUNTIME_TOTAL_EXECUTOR_CORES

if [ -z "${KMP_AFFINITY}" ]; then
    export KMP_AFFINITY=granularity=fine,compact,1,0
fi

if [ -z "${OMP_NUM_THREADS}" ]; then
    if [ -z "${ZOO_NUM_MKLTHREADS}" ]; then
        export OMP_NUM_THREADS=1
    else
        if [ `echo $ZOO_NUM_MKLTHREADS | tr '[A-Z]' '[a-z]'` == "all" ]; then
            export OMP_NUM_THREADS=`nproc`
        else
            export OMP_NUM_THREADS=${ZOO_NUM_MKLTHREADS}
        fi
    fi
fi

if [ -z "${KMP_BLOCKTIME}" ]; then
    export KMP_BLOCKTIME=0
fi

# verbose for OpenMP
if [[ $* == *"verbose"* ]]; then
    export KMP_SETTINGS=1
    export KMP_AFFINITY=${KMP_AFFINITY},verbose
fi

${SPARK_HOME}/bin/pyspark \
  --master ${RUNTIME_SPARK_MASTER} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --conf spark.driver.host=localhost \
  --conf spark.pyspark.python=python3 \
  --conf spark.pyspark.driver.python=ipython3 \
  --conf spark.executorEnv.TF_DISABLE_MKL=1 \
  --conf spark.executor.memoryOverhead=2048 \
  --conf spark.driver.memoryOverhead=1024 \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --jars ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory'
