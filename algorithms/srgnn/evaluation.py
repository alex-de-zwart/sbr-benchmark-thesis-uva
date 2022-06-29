import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from algorithms.srgnn.model import *
# from model import *

DATA_DIR = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/srgnn'
MODEL_DIR = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn/trained_models/'
TRAIN_DATA = '/home/ec2-user/SageMaker/sb-rec-system/data/prepared/vsknn'



CONFIG = {
#   "batch_size": 64,
  "batch_size": 1,
  "epoch": 10,
  "hidden_size": 125,
  "l2": 1e-05,
  "lr": 0.0007160251445470053,
  "lr_dc": 0.1,
  "step": 1
}

if __name__ == "__main__":

    data_dir = DATA_DIR
    model_dir = MODEL_DIR
    config = CONFIG
    train_dir = TRAIN_DATA

    for data_slice in range(3):
        latencies_out_file = f'latencies/latencies_{data_slice}.txt'
        pred_out_file = f'predictions/predictions_{data_slice}.txt'
        train_file = f'item_views_train.{data_slice}.txt'
        reversed_item_dict = f'item_views_train.{data_slice}.reversed_item_dict.pkl'
        
        _, test_data, test_data_original, n_nodes = load_data(data_dir, data_slice, train=False)
        
        #read original training file
        train_data = pd.read_csv(train_dir + os.sep + train_file, sep='\t')

        #use item dict
        with open(data_dir + os.sep + reversed_item_dict, 'rb') as rvdict:
            item_dict = pickle.load(rvdict)

        train_data.replace({"ItemId": item_dict})
        _, test_data, test_data_original, n_nodes = load_data(data_dir, data_slice, train=False)


        model = trans_to_cuda(SessionGraph(config, n_nodes))
        path = os.path.join(model_dir, f"model_{data_slice}")
        model.load_state_dict(torch.load(path))
        test_score(train_data, model, test_data, test_data_original, latencies_out_file, pred_out_file)