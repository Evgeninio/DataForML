
from dagster import job
from assets.collect_data import collect_sms_data_op
from assets.preprocess_and_entropy import preprocess_and_entropy_op
from assets.clean_data import clean_data_op

@job
def data_collection_job():
    df = collect_sms_data_op()
    df2 = preprocess_and_entropy_op(df)
    clean_data_op(df2)
