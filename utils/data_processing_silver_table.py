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

def process_feature_silver_table(snapshot_date_str, bronze_feature_directory, silver_feature_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # feature tables
    def load_csv(name):
        filename = f"bronze_{name}_{snapshot_date_str.replace('-', '_')}.csv"
        path = os.path.join(bronze_feature_directory, filename)
        print(f"Loading {name} from: {path}")
        return spark.read.csv(path, header=True, inferSchema=True)

    # load bronze-level feature data
    df_clickstream = load_csv("clickstream")
    print("row count: ", df_clickstream.count())
    df_attributes = load_csv("attributes")
    print("row count: ", df_attributes.count())
    df_financials = load_csv("financials")
    print("row count: ", df_financials.count())

    # data cleaning --------------------------------------------------------------------------
    # clickstream
    # handle missing vlaues
    df_clickstream = df_clickstream.na.fill({
        **{f"fe_{i}": 0 for i in range(1, 21)}
    })

    # attributes
    # Age
    # remove underscore
    df_attributes = df_attributes.withColumn("Age", F.regexp_replace("Age", "_", ""))
    # case to integer
    df_attributes = df_attributes.withColumn("Age", col("Age").cast("int"))
    # replace invalid values with None
    df_attributes = df_attributes.withColumn(
        "Age",
        F.when((col("Age") < 0) | (col("Age") > 120), None).otherwise(col("Age"))
    )
    # fill missing vlaues with median
    median_age = df_attributes.approxQuantile("Age", [0.5], 0.01)[0]
    df_attributes = df_attributes.na.fill({"Age": median_age})

    # SSN
    valid_ssn_regex = r"^\d{3}-\d{2}-\d{4}$" # pattern of valid SSN
    # replace invalid SSN with null
    df_attributes = df_attributes.withColumn(
        "SSN",
        F.when(col("SSN").rlike(valid_ssn_regex), col("SSN")).otherwise("Unkonwn")
    )

    # Occupation
    invalid_occupations = ["_______", ""]
    df_attributes = df_attributes.withColumn(
        "Occupation",
        F.when(F.trim(col("Occupation")).isin(invalid_occupations), "Unkonwn").otherwise(col("Occupation"))
    )

    # financials
    # numerical
    # Annual_Income
    # remove underscore
    df_financials = df_financials.withColumn(
        "Annual_Income",
        F.regexp_replace(col("Annual_Income"), "_", "").cast("float")
    )

    # Num_Bank_Accounts
    # fill with median
    median_num_bank_accounts = df_financials.approxQuantile("Num_Bank_Accounts", [0.5], 0.01)[0]
    percentile_99 = df_financials.approxQuantile("Num_Bank_Accounts", [0.99], 0.01)[0]
    df_financials = df_financials.withColumn(
        "Num_Bank_Accounts",
        F.when(col("Num_Bank_Accounts") > percentile_99, median_num_bank_accounts).otherwise(col("Num_Bank_Accounts"))
    )

    # Num_Credit_Card
    # fill with median
    median_num_credit_card = df_financials.approxQuantile("Num_Credit_Card", [0.5], 0.01)[0]
    percentile_99 = df_financials.approxQuantile("Num_Credit_Card", [0.99], 0.01)[0]
    df_financials = df_financials.withColumn(
        "Num_Credit_Card",
        F.when(col("Num_Credit_Card") > percentile_99, median_num_bank_accounts).otherwise(col("Num_Credit_Card"))
    )

    # Interest_Rate
    df_financials = df_financials.withColumn(
        "Interest_Rate",
        F.when((col("Interest_Rate") < 0) | (col("Interest_Rate") > 100), None).otherwise(col("Interest_Rate"))
    )
    # fill missing vlaues with median
    median_interest_rate = df_financials.approxQuantile("Interest_Rate", [0.5], 0.01)[0]
    df_financials = df_financials.na.fill({"Interest_Rate": median_interest_rate})

    # Num_of_Loan
    # remove underscore
    df_financials = df_financials.withColumn("Num_of_Loan", F.regexp_replace("Num_of_Loan", "_", ""))
    # case to integer
    df_financials = df_financials.withColumn("Num_of_Loan", col("Num_of_Loan").cast("int"))
    # replace invalid values with None
    df_financials = df_financials.withColumn(
        "Num_of_Loan",
        F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 100), None).otherwise(col("Num_of_Loan"))
    )
    # fill missing vlaues with median
    median_num_of_loan = df_financials.approxQuantile("Num_of_Loan", [0.5], 0.01)[0]
    df_financials = df_financials.na.fill({"Num_of_Loan": median_num_of_loan})

    # Num_of_Delayed_Payment
    # remove underscore
    df_financials = df_financials.withColumn("Num_of_Delayed_Payment", F.regexp_replace("Num_of_Delayed_Payment", "_", ""))
    # case to integer
    df_financials = df_financials.withColumn("Num_of_Delayed_Payment", col("Num_of_Delayed_Payment").cast("int"))
    # replace invalid values with None
    df_financials = df_financials.withColumn(
        "Num_of_Delayed_Payment",
        F.when((col("Num_of_Delayed_Payment") < 0) | (col("Num_of_Delayed_Payment") > 100), None).otherwise(col("Num_of_Delayed_Payment"))
    )
    # fill missing vlaues with median
    median_num_of_delayed_payment = df_financials.approxQuantile("Num_of_Delayed_Payment", [0.5], 0.01)[0]
    df_financials = df_financials.na.fill({"Num_of_Delayed_Payment": median_num_of_delayed_payment})

    # Num_Credit_Inquiries
    df_financials = df_financials.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast("int"))
    # replace invalid values with None
    df_financials = df_financials.withColumn(
        "Num_Credit_Inquiries",
        F.when((col("Num_Credit_Inquiries") < 0) | (col("Num_Credit_Inquiries") > 100), None).otherwise(col("Num_Credit_Inquiries"))
    )
    # fill missing vlaues with median
    median_num_of_credit_inquiries = df_financials.approxQuantile("Num_Credit_Inquiries", [0.5], 0.01)[0]
    df_financials = df_financials.na.fill({"Num_Credit_Inquiries": median_num_of_credit_inquiries})

    # Outstanding_Debt
    # remove underscore
    df_financials = df_financials.withColumn("Outstanding_Debt", F.regexp_replace("Outstanding_Debt", "_", ""))

    # Amount_invested_monthly
    # remove underscore
    df_financials = df_financials.withColumn("Amount_invested_monthly", F.regexp_replace("Amount_invested_monthly", "_", ""))

    # categorical
    # Type_of_Loan
    df_financials = df_financials.withColumn(
        "Type_of_Loan",
        F.when(col("Type_of_Loan").isNull(), "Unknown").otherwise(col("Type_of_Loan"))
    )

    # Credit_Mix
    df_financials = df_financials.withColumn(
        "Credit_Mix",
        F.when(col("Credit_Mix") == "_", "Unknown").otherwise(col("Credit_Mix"))
    )

    # Payment_of_Min_Amount
    df_financials = df_financials.withColumn(
        "Payment_of_Min_Amount",
        F.when(col("Payment_of_Min_Amount") == "NM", "Unknown").otherwise(col("Payment_of_Min_Amount"))
    )

    # Payment_Behaviour
    valid_payment_behaviour_regex = r"^[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+_[a-zA-Z]+$"
    # replace invalid values with null
    df_financials = df_financials.withColumn(
        "Payment_Behaviour",
        F.when(col("Payment_Behaviour").rlike(valid_payment_behaviour_regex), col("Payment_Behaviour")).otherwise("Unkonwn")
    )

    # schema definitions -----------------------------------------------------------------------
    clickstream_schema = {
        **{f"fe_{i}": IntegerType() for i in range(1, 21)},
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    attributes_schema = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    financials_schema = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    def enforce_schema(df, schema_dict):
        for col_name, dtype in schema_dict.items():
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast(dtype))
        return df

    df_clickstream = enforce_schema(df_clickstream, clickstream_schema)
    df_attributes = enforce_schema(df_attributes, attributes_schema)
    df_financials = enforce_schema(df_financials, financials_schema)

    # save to csv
    for df, name in zip([df_clickstream, df_attributes, df_financials], ['clickstream', 'attributes', 'financials']):

        partition_name = f"silver_{name}_{snapshot_date_str.replace('-', '_')}.parquet"
        filepath = silver_feature_directory + partition_name
        df.write.mode("overwrite").parquet(filepath)
        print(f'Saved {name} to:', filepath)

def process_label_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df