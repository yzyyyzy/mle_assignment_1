import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_feature_bronze_table(snapshot_date_str, bronze_feature_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # define the input CSV paths
    clickstream_path = "data/feature_clickstream.csv"
    attributes_path = "data/features_attributes.csv"
    financials_path = "data/features_financials.csv"
    
    # read each CSV
    df_clickstream = spark.read.csv(clickstream_path, header=True, inferSchema=True)
    df_attributes = spark.read.csv(attributes_path, header=True, inferSchema=True)
    df_financials = spark.read.csv(financials_path, header=True, inferSchema=True)

    # filter each DataFrame by snapshot_date if the column exists
    for df, name in zip([df_clickstream, df_attributes, df_financials], ['clickstream', 'attributes', 'financials']):
        if 'snapshot_date' in df.columns:
            df = df.filter(col('snapshot_date') == snapshot_date)
        
        print(f'{name} - {snapshot_date_str} row count:', df.count())

        # save to CSV
        partition_name = f"bronze_{name}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(bronze_feature_directory, partition_name)
        df.toPandas().to_csv(filepath, index=False)
        print(f'Saved {name} to:', filepath)

def process_label_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)