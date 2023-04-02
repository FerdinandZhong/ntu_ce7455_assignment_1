import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.constants import EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM
from utils.data import DataProcessing
from utils.models import RNN, MLP, CNN, LSTM
from utils.utils import epoch_time, evaluate, register_logger, train
from utils.utils import count_parameters
import argparse


logger = register_logger("experiments/task5.log")
model_mappings = {
    "mlp": MLP,
    "cnn": CNN,
    "lstm": LSTM
}


def t5_trainer(
    epochs,
    model,
    optimizer,
    train_iterator,
    valid_iterator,
    criterion,
    output_model_name,
    tensorboard_log_dir,
):
    logger.info(f"Number of parameters: {count_parameters(model):,} trainable parameters")

    N_EPOCHS = epochs

    tensorboard_writter = SummaryWriter(tensorboard_log_dir)

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_model_name)

        logger.info(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        logger.info(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

        tensorboard_writter.add_scalar("Epoch Loss/train", train_loss, epoch + 1)
        tensorboard_writter.add_scalar("Epoch Acc/train", train_acc, epoch + 1)
        tensorboard_writter.add_scalar("Epoch Loss/valid", valid_loss, epoch + 1)
        tensorboard_writter.add_scalar("Epoch Acc/valid", valid_acc, epoch + 1)
        for tag, value in model.named_parameters():
            if value.grad is not None:
                tensorboard_writter.add_histogram(
                    tag + "/grad", value.grad.cpu(), epoch + 1
                )


def t5_evaluator(model, output_model_name, test_iterator, criterion):
    model.load_state_dict(torch.load(output_model_name))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    logger.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mlp")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.25)
    opt = parser.parse_args()
    device = torch.device("cuda:0")
    criterion = torch.nn.BCEWithLogitsLoss()

    all_data = DataProcessing()
    train_iterator, valid_iterator, test_iterator = all_data.generate_iterator(device)

    with open("experiments/task5.log", "a+") as file:
        file.write("\n-------------------------------------------\n")
    
    if opt.model_type.lower() == "mlp":
        logger.info(
            f"Starting the experiment for task 5 with MLP"
        )
        hidden_dim_layers_list = [[500], [500, 300], [500, 300, 200]]
        for hidden_dim_layers in hidden_dim_layers_list:
            logger.info(f"Hidden dims: {hidden_dim_layers}")
            model = MLP(INPUT_DIM, EMBEDDING_DIM, hidden_dim_layers, OUTPUT_DIM).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
            t5_trainer(
                epochs=50, 
                model=model,
                optimizer=optimizer,
                train_iterator=train_iterator,
                valid_iterator=valid_iterator,
                criterion=criterion,
                output_model_name=f"models/t5/mlp/mlp_{len(hidden_dim_layers)}_layers.pt",
                tensorboard_log_dir=f"logs/t5/mlp/mlp_{len(hidden_dim_layers)}_layers"
            )
            t5_evaluator(
                model,
                f"models/t5/mlp/mlp_{len(hidden_dim_layers)}_layers.pt",
                test_iterator,
                criterion
            )
    elif opt.model_type.lower() == "cnn":
        logger.info(
            f"Starting the experiment for task 5 with CNN, lr: {opt.lr}"
        )
        model = CNN(INPUT_DIM, EMBEDDING_DIM, [1,2,3], HIDDEN_DIM, OUTPUT_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        t5_trainer(
            epochs=50, 
            model=model,
            optimizer=optimizer,
            train_iterator=train_iterator,
            valid_iterator=valid_iterator,
            criterion=criterion,
            output_model_name="models/t5/cnn/cnn.pt",
            tensorboard_log_dir=f"logs/t5/cnn_{opt.lr}"
        )
        t5_evaluator(
            model,
            "models/t5/cnn/cnn.pt",
            test_iterator,
            criterion
        )
    elif opt.model_type.lower() == "lstm":
        logger.info(
            f"Starting the experiment for task 5 with LSTM, lr: {opt.lr}, dropout rate: {opt.dropout_rate}"
        )
        model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, dropout_rate=opt.dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        t5_trainer(
            epochs=50, 
            model=model,
            optimizer=optimizer,
            train_iterator=train_iterator,
            valid_iterator=valid_iterator,
            criterion=criterion,
            output_model_name="models/t5/lstm/lstm.pt",
            tensorboard_log_dir=f"logs/t5/lstm_{opt.lr}"
        )
        t5_evaluator(
            model,
            "models/t5/lstm/lstm.pt",
            test_iterator,
            criterion
        )
    elif opt.model_type.lower() == "bilstm":
        logger.info(
            f"Starting the experiment for task 5 with Bi-directional LSTM, lr: {opt.lr}, dropout rate: {opt.dropout_rate}"
        )
        model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, is_bidirectional=True, dropout_rate=opt.dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        t5_trainer(
            epochs=50, 
            model=model,
            optimizer=optimizer,
            train_iterator=train_iterator,
            valid_iterator=valid_iterator,
            criterion=criterion,
            output_model_name="models/t5/bi-lstm/bi-lstm.pt",
            tensorboard_log_dir=f"logs/t5/bi-lstm_{opt.lr}"
        )
        t5_evaluator(
            model,
            "models/t5/bi-lstm/bi-lstm.pt",
            test_iterator,
            criterion
        )
    else:
        logger.info(
            f"Record all model parameters count"
        )
        hidden_dim_layers_list = [[500], [500, 300], [500, 300, 200]]
        for hidden_dim_layers in hidden_dim_layers_list:
            model = MLP(INPUT_DIM, EMBEDDING_DIM, hidden_dim_layers, OUTPUT_DIM).to(device)
            logger.info(f"Number of parameters: {count_parameters(model):,} trainable parameters")
        model = CNN(INPUT_DIM, EMBEDDING_DIM, [1,2,3], HIDDEN_DIM, OUTPUT_DIM).to(device)
        logger.info(f"Number of parameters: {count_parameters(model):,} trainable parameters")
