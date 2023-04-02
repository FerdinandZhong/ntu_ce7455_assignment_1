import argparse
import time

import gensim
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.constants import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM
from utils.data import DataProcessing, generate_pretraining_embedding_vectors
from utils.models import RNN, LSTM, CNN
from utils.utils import epoch_time, evaluate, register_logger, train, count_parameters
from functools import partial

logger = register_logger("experiments/task4.log")

model_dict = {
    "rnn": RNN,
    "cnn": CNN,
    "lstm": LSTM,
    "bilstm": partial(LSTM, is_bidirectional=True)
}

def t4_trainer(
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


def t4_evaluator(model, output_model_name, test_iterator, criterion):
    model.load_state_dict(torch.load(output_model_name))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    logger.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="rnn")
    parser.add_argument("--pretraining_type", type=str, default="word2vec")
    parser.add_argument(
        "--pretraining_model", type=str, default="GoogleNews-vectors-negative300.bin.gz"
    )
    parser.add_argument(
        "--model_output_path", type=str, default="models/t4/adam_pretrained_word2vec.pt"
    )
    parser.add_argument("--tensorboard_dir", type=str, default="logs/t4/word2vec")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_emb", action="store_true")
    opt = parser.parse_args()
    device = torch.device("cuda:0")
    criterion = torch.nn.BCEWithLogitsLoss()

    with open("experiments/task4.log", "a+") as file:
        file.write("\n-------------------------------------------\n")


    if "word2vec" in opt.pretraining_type:
        all_data = DataProcessing()
        train_iterator, valid_iterator, test_iterator = all_data.generate_iterator(
            device
        )
        pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(
            opt.pretraining_model, binary=True
        )
        pretrained_embedding_vectors = generate_pretraining_embedding_vectors(
            pretrained_model, all_data.text.vocab, logger
        )
    else:
        all_data = DataProcessing(pretrained=opt.pretraining_model)
        train_iterator, valid_iterator, test_iterator = all_data.generate_iterator(
            device
        )
        pretrained_embedding_vectors = all_data.text.vocab.vectors

    
    logger.info(
        f"Starting the experiment for task 4 with model: {opt.model_type} pretraining: {opt.pretraining_type} lr: {opt.lr}"
    )
    logger.info("Training")
    embedding_dim = pretrained_embedding_vectors.shape[1]

    freeze = opt.freeze_emb if opt.freeze_emb else False
    model = model_dict[opt.model_type](
        INPUT_DIM,
        embedding_dim,
        HIDDEN_DIM,
        OUTPUT_DIM,
        pretrained_weight = pretrained_embedding_vectors,
        pretraining_freeze=freeze,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    t4_trainer(
        50,
        model,
        optimizer,
        train_iterator,
        valid_iterator,
        criterion,
        f"models/t4/{opt.model_output_path}",
        f"logs/t4/{opt.tensorboard_dir}",
    )
    logger.info("Testing")
    t4_evaluator(
        model, f"models/t4/{opt.model_output_path}", test_iterator, criterion=criterion
    )
