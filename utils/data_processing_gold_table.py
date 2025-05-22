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
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

def process_feature_gold_table(snapshot_date_str, silver_feature_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    def load_parquet(silver_dir, table_name):
        path = os.path.join(silver_dir, f"silver_{table_name}_{snapshot_date_str.replace("-", "_")}.parquet")
        print(path)
        return spark.read.parquet(path)

    df_clickstream = load_parquet(silver_feature_directory, 'clickstream')
    # get all the attributes data
    df_attributes = spark.read.parquet(silver_feature_directory + "silver_attributes_*.parquet")
    df_attributes = df_attributes.drop('snapshot_date')
    # get all the financials data
    df_financials = spark.read.parquet(silver_feature_directory + "silver_financials_*.parquet")
    df_financials = df_financials.drop('snapshot_date')
    # join the tables
    df_features = df_clickstream.join(df_attributes, on=["Customer_ID"], how='inner') \
                                .join(df_financials, on=["Customer_ID"], how='inner')

    # -----------------------------------------------------------------------------------------------
    # feature engineering ---------------------------------------------------------------------------
    stages = []
    # categorical columns
    # ont-hot encoding for categorical columns
    categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # create column for each type of loan
    type_of_loans = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Not Specified', 'Mortgage Loan', 'Payday Loan', 'Student Loan', 'Personal Loan']
    for loan_type in type_of_loans:
        col_name = loan_type.replace(" ", "_").replace("-", "_")
        df_features = df_features.withColumn(
            f"has_{col_name}",
            F.when(F.col("Type_of_Loan").contains(loan_type), F.lit("Yes")).otherwise(F.lit("No"))
        )
    transformed_loans = ['has_' + loan.replace(' ', '_').replace('-', '_') for loan in type_of_loans]

    categorical_cols.extend(transformed_loans)
    # one-hot encoding
    for cat_col in categorical_cols:
        indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_index", handleInvalid="keep")
        df_features = indexer.fit(df_features).transform(df_features)

        encoder = OneHotEncoder(inputCol=f"{cat_col}_index", outputCol=f"{cat_col}_ohe")
        df_features = encoder.fit(df_features).transform(df_features)

        df_features = df_features.drop(cat_col).drop(f"{cat_col}_index")

    # convert Credit_History_Age to numbe of month
    df_features = df_features.withColumn(
        "Years",
        F.regexp_extract("Credit_History_Age", r"(\d+)\s+Years?", 1).cast("int")
    ).withColumn(
        "Months",
        F.regexp_extract("Credit_History_Age", r"(\d+)\s+Months?", 1).cast("int")
    )
    df_features = df_features.withColumn(
        "Credit_History_Month",
        df_features["Years"] * 12 + df_features["Months"]
    ) # will be treated as numerical column later
    df_features = df_features.drop("Years", "Months", "Credit_History_Age")
    
    # numerical columns
    numerical_cols = ['fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5', 'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10', 'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15', 'fe_16', 'fe_17', 'fe_18', 'fe_19', 'fe_20', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Month']

    # normalization
    assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numeric_features", handleInvalid="skip")
    df_features = assembler.transform(df_features)
    scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
    scaler_model = scaler.fit(df_features)
    df_features = scaler_model.transform(df_features)

    # drop unnecessary columns
    df_features = df_features.drop('Name', 'SSN', 'Type_of_Loan')

    df_features.show()

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_features.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df_features

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df