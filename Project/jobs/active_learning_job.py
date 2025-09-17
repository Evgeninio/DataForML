
from dagster import job
from assets.active_learning import active_learning_op
from assets.train_model import train_model_op
from assets.preprocess_and_entropy import preprocess_and_entropy_op
from assets.collect_data import collect_sms_data_op
from assets.clean_data import clean_data_op

@job
def active_learning_job():
    raw = collect_sms_data_op()
    with_entropy = preprocess_and_entropy_op(raw)
    clean = clean_data_op(with_entropy)
    model = active_learning_op(clean)
    train_model_op(model, with_entropy)
