#!/usr/bin/python
import argparse
import importlib
import time
import datetime
import sys
import os
import math

from zoo.orca import init_orca_context, OrcaContext
from zoo.orca.learn.tf2.estimator import Estimator
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType


if os.path.exists('jobs.zip'):
    sys.path.insert(0, 'jobs.zip')
else:
    sys.path.insert(0, './jobs')

# load dynamic module
ncf_features = importlib.import_module("jobs.ncf_features")
ncf_model = importlib.import_module("jobs.ncf_model")


__author__ = 'e047349'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a NCF DeepLearning PySpark job')
    parser.add_argument("-n","--app_name",help="app name",type=str,required=True)
    parser.add_argument("-d","--data_source_path",help="data path of data source",type=str,required=True)
    parser.add_argument("-m","--model_dir", help="location to save model output",type=str,required=True)
    parser.add_argument("-u","--u_limit", help="target amount of users",type=str,required=True)
    parser.add_argument("-i","--m_limit", help="target amount of items",type=str,required=True)
    parser.add_argument("-r","--neg_rate", help="rate of generate negitive samples ",type=str,required=True)
    parser.add_argument("-s","--sliding_length", help="Length of sliding window",required=True)
    parser.add_argument("-o","--u_output", help="Output of users embedding",required=True)
    parser.add_argument("-t","--m_output", help="Output of items embedding",required=True)
    parser.add_argument("-e","--max_epoch", help="max epoch to run",required=True)
    parser.add_argument("-b","--batch_size", help="batch size for each iteration",required=True)
    parser.add_argument("-l","--log_dir", help="location of log file",required=True)
    parser.add_argument("-as","--train_start", help="start date of training",required=True)
    parser.add_argument("-ae","--train_end", help="end date of training",required=True)
    parser.add_argument("-vs","--validation_start", help="start date of validation",required=True)
    parser.add_argument("-ve","--validation_end", help="end date of validation",required=True)
    parser.add_argument("-ts","--test_start", help="start date of test",required=True)
    parser.add_argument("-te","--test_end", help="end date of test",required=True)
    parser.add_argument("-is","--inference_start", help="start date of inference",required=True)
    parser.add_argument("-ie","--inference_end", help="end date of inference",required=True)
    parser.add_argument("-io","--inference_output_path", help="output folder of inference",required=True)
    args = parser.parse_args()
    print(args)
    app_name = args.app_name
    data_source_path = args.data_source_path
    model_file_name = app_name + '.h5'
    save_model_file = args.model_dir + model_file_name
    u_limit = int(args.u_limit)
    m_limit = int(args.m_limit)
    neg_rate = int(args.neg_rate)
    sliding_length = int(args.sliding_length)
    u_output = int(args.u_output)
    m_output = int(args.m_output)
    max_epoch = int(args.max_epoch)
    batch_size = int(args.batch_size)
    predict_output_path = args.inference_output_path

    sc = init_orca_context(cluster_mode="spark-submit", init_ray_on_spark=True)
    spark = OrcaContext.get_spark_session()

    start = time.time()
    uDF, mDF, tDF = ncf_features.load_csv(spark,data_source_path,u_limit,m_limit)
    trainingDF = ncf_features.genData(tDF,sc,spark,args.train_start, args.train_end,neg_rate,sliding_length,u_limit,m_limit)
    #trainingDF.show(5)
    validationDF = ncf_features.genData(tDF,sc,spark,args.validation_start, args.validation_end,neg_rate,sliding_length,u_limit,m_limit)
    validationDF.show(5)
    testDF = ncf_features.genData(tDF,sc,spark,args.test_start,args.test_end,neg_rate,sliding_length,u_limit,m_limit)
    #testDF.show(5)
    inferenceDF = ncf_features.genData(tDF,sc,spark,args.inference_start,args.inference_end,neg_rate,sliding_length,u_limit,m_limit)
    #inferenceDF.show(5)

    # model = ncf_model.getKerasModel(u_limit,m_limit,u_output,m_output,args.log_dir)
    def get_model(config):
        return ncf_model.getKerasModel(u_limit,m_limit,u_output,m_output,args.log_dir)
    est = Estimator.from_keras(model_creator=get_model, model_dir=args.log_dir)
    steps = math.ceil(trainingDF.count() / batch_size)
    valid_steps = math.ceil(validationDF.count() / batch_size)
    est.fit(data=trainingDF,
            batch_size=batch_size,
            epochs=max_epoch,
            steps_per_epoch=steps,
            feature_cols=['features'],
            label_cols=['labels'],
            validation_data=validationDF,
            validation_steps=valid_steps
            )
    # save checkpoint
    est.save(os.path.join(args.model_dir, "ncf.ckpt"))
    # metrics ,result and save keras model
    model = est.get_model()
    model.save(save_model_file)
    print(model.metrics_names)
    #Orca the predict function supports native spark data frame ! Just need to tell batch_size and feature_cols
    # use a new Estimamtor to validate load model API
    est.load(os.path.join(args.model_dir, "ncf.ckpt"))
    prediction_df = est.predict(inferenceDF, batch_size=batch_size, feature_cols=['features'])
    prediction_df.show(5)
    score_udf = udf(lambda pred: 0.0 if pred[0] > pred[1] else 1.0, FloatType())
    prediction_df = prediction_df.withColumn('prediction2', score_udf('prediction'))
    prediction_df.show(10)
    # Save Table
    #prediction_final_df.write.mode('overwrite').parquet(predict_output_path)
    prediction_df.select('uid','mid','prediction2').write.mode('overwrite').parquet(predict_output_path)
    #prediction_df.select('uid','mid','prediction2').write.mode('overwrite').format("csv").save(predict_output_path)
    #user_join_df = prediction_df.join(uDF, on=['uid'], how='inner')
    #prediction_final_df = user_join_df.join(mDF, on=['mid'], how='inner').select('u','m','prediction').write.mode('overwrite').parquet(predict_output_path)
    end = time.time()
    print("Took time:"+str((end-start)))
