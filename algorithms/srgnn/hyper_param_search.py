import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from model import *

# DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..', 'data\prepared\srgnn'))
#aws dir
DATA_DIR = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/srgnn/'
CHECKPOINT_DIR = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn/ray/'

def hyper_settings(config, num_samples=50, max_num_epochs=10, gpus_per_trial=0):
    data_dir = DATA_DIR
    checkpoint_dir = CHECKPOINT_DIR
    scheduler = ASHAScheduler(
        metric="mrr",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        max_progress_rows=10,
        metric="mrr",
        mode="max",
        metric_columns=["loss", "hit", "mrr"])
    result = tune.run(
        partial(train_hyper, data_dir=data_dir,checkpoint_dir=checkpoint_dir ),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("mrr", " max", "last")
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial final validation loss : {best_trial.last_result["loss"]}')
    print(f'Best trial final validation MRR@20: {best_trial.last_result["mrr"]}')



if __name__ == "__main__":

    # config for hyper param search
    config = {
        "l2": 1e-5,
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 100, 128, 256]),
        "hidden_size": tune.choice([50, 100, 125, 150, 200]),
        "step": tune.choice([1, 2, 3]),
        "lr_dc": 0.1,
        "epoch": 10
    }

    hyper_settings(config)





