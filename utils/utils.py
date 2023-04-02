import logging

import torch


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, _ = batch.text

        predictions = model(text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, _ = batch.text

            predictions = model(text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatter = (
        "%(asctime)s - {color}%(levelname)s{reset} - %(filename)s:%(lineno)d"
        " - %(module)s.%(funcName)s - %(process)d - %(message)s"
    )
    FORMATS = {
        logging.DEBUG: formatter.format(color=grey, reset=reset),
        logging.INFO: formatter.format(color=green, reset=reset),
        logging.WARNING: formatter.format(color=yellow, reset=reset),
        logging.ERROR: formatter.format(color=red, reset=reset),
        logging.CRITICAL: formatter.format(color=bold_red, reset=reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def register_logger(logging_path):
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logging_path)
    file_handler.setFormatter(
        logging.Formatter(
            (
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d"
                " - %(module)s.%(funcName)s - %(process)d - %(message)s"
            ),
            "%Y-%m-%dT%H:%M:%S.%f%z",
        )
    )
    logger.addHandler(file_handler)

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(ColorfulFormatter())
    logger.addHandler(streaming_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)