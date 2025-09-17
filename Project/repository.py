
from dagster import Definitions
from assets.collect_data import collect_sms_data_op
from assets.preprocess_and_entropy import preprocess_and_entropy_op
from assets.clean_data import clean_data_op
from assets.active_learning import active_learning_op
from assets.train_model import train_model_op

from jobs.data_collection_job import data_collection_job
from jobs.active_learning_job import active_learning_job

defs = Definitions(
    jobs=[data_collection_job, active_learning_job],
    assets=[],
    resources={},
)
