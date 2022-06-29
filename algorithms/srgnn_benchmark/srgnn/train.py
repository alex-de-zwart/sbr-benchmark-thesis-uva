import sys
module_path = "/home/ec2-user/SageMaker/sb-rec-system"
if module_path not in sys.path:
    sys.path.append(module_path)

from algorithms.srgnn_benchmark.srgnn.model import *


def train(config):
    model = SRGNN(config["hidden_dim"], config["num_items"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 weight_decay=config["l2_penalty"])

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config["step"],
                                          gamma=config["weight_decay"])
    
    train_dataset = GraphDataset('/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn_benchmark/data/diginetica', 'train')


    train_loader = pyg_data.DataLoader(train_dataset,
                                       batch_size=config["batch_size"],
                                       shuffle=False,
                                       drop_last=False)
    
    val_dataset = GraphDataset('/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn_benchmark/data/diginetica', 'test')


    val_loader = pyg_data.DataLoader(val_dataset,
                                     batch_size=config["batch_size"],
                                     shuffle=False,
                                     drop_last=False)
    # Train
    losses = []
    test_accs = []
    test_mrr = []
    top_k_accs = []
    top_k_mrrs = []

    best_acc = 0
    best_model = None

    for epoch in range(config["epochs"]):
        total_loss = 0
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            batch.to('cpu')
            optimizer.zero_grad()

            pred = model(batch)
            label = batch.y
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(train_loader.dataset)
        losses.append(total_loss)

        scheduler.step()

    checkpoint_dir = '/home/ec2-user/SageMaker/sb-rec-system/algorithms/srgnn_benchmark/trained_models'
    path = os.path.join(checkpoint_dir, f"srgnn_diginetica")
    torch.save(model.state_dict(), path)
    
  

if __name__ == "__main__":
    #best config settings from hyper param search
    config = {
        'l2_penalty': 1e-05, 
        'lr': 0.001, 
        'epochs': 10, 
        'batch_size': 100, 
        'hidden_dim': 100, 
        'step': 3, 
        'weight_decay': 0.1,
        'num_items': 43098
    }

    train(config)