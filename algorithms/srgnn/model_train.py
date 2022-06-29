import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from algorithms.srgnn.model import *

#best config settings from hyper param search
CONFIG = {
  "batch_size": 64,
  "epoch": 10,
  "hidden_size": 125,
  "l2": 1e-05,
  "lr": 0.0007160251445470053,
  "lr_dc": 0.1,
  "step": 1
}

DATA_DIR = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/srgnn'


if __name__ == "__main__":
    data_dir = DATA_DIR
    config = CONFIG

    for data_slice in range(3):
        train_data, test_data, test_data_original, n_node = load_data(data_dir, data_slice, train=False)

        # train and write model
        train(config, train_data, n_node, data_slice, checkpoint_dir=None, data_dir=None)