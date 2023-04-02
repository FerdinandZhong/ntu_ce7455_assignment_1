import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.constants import EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM
from utils.data import DataProcessing
from utils.models import RNN
from utils.utils import epoch_time, evaluate, register_logger, train
from utils.utils import epoch_time, evaluate, register_logger, train, count_parameters
import os

logger = register_logger("experiments/task3.log")


def t3_trainer(
    epochs,
    model,
    optimizer,
    train_iterator,
    valid_iterator,
    criterion,
    output_model_name,
    tensorboard_log_dir,
):
    N_EPOCHS = epochs
    logger.info(f"Number of parameters: {count_parameters(model):,} trainable parameters")
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


def t3_evaluator(model, output_model_name, test_iterator, criterion):
    model.load_state_dict(torch.load(output_model_name))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    logger.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")


with open("experiments/task4.log", "a+") as file:
    file.write("\n-------------------------------------------\n")


all_data = DataProcessing()
device = torch.device("cuda:0")
train_iterator, valid_iterator, test_iterator = all_data.generate_iterator(device)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch_num in [5, 10, 20, 50]:
    logger.info(f"Starting the experiment for epochs: {epoch_num} with dropout")
    logger.info("Training")
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, dropout_rate=float(os.getenv("dropout_rate"))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    t3_trainer(
        epoch_num,
        model,
        optimizer,
        train_iterator,
        valid_iterator,
        criterion,
        f"models/t3/adam_epochs_{epoch_num}.pt",
        f"logs/t3/epochs_{epoch_num}",
    )
    logger.info("Testing")
    t3_evaluator(
        model,
        f"models/t3/adam_epochs_{epoch_num}.pt",
        test_iterator,
        criterion=criterion,
    )
