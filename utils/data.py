import random

import torch
from torchtext import data, datasets
# import gensim


class DataProcessing:
    def __init__(self, max_vocab_size=25000, pretrained=None) -> None:
        SEED = 1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.text = data.Field(
            tokenize="spacy",
            tokenizer_language="en_core_web_sm",
            include_lengths=True,
            pad_first=True,
        )

        self.label = data.LabelField(dtype=torch.float)

        train_data, self.test_data = datasets.IMDB.splits(self.text, self.label)
        """
        dataset in torchtext is the numerical one already, build vocab with max_size will 
        define how many tokens to be kept.
        load_vectors is using the itos to pick the correct vector embeddings for the token
        """

        self.train_data, self.valid_data = train_data.split(
            random_state=random.seed(SEED)
        )

        if not pretrained:
            self.text.build_vocab(self.train_data, max_size=max_vocab_size)
            self.label.build_vocab(self.train_data)
        else:
            self.text.build_vocab(
                self.train_data, max_size=max_vocab_size, vectors=pretrained
            )
            self.label.build_vocab(self.train_data)

    def generate_iterator(self, device, batch_size=64):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=batch_size,
            sort_within_batch=True,
            device=device,
        )
        return train_iterator, valid_iterator, test_iterator


def generate_pretraining_embedding_vectors(pretrained_model, vocab, logger):
    vectors = []
    total_unknown_tokens = 0
    for token in vocab.itos:
        token = token.replace("<", "").replace(">", "")
        if token in pretrained_model.vocab:
            vectors.append(torch.from_numpy(pretrained_model[token]))
        else:
            vectors.append(torch.from_numpy(pretrained_model["unk"]))
            total_unknown_tokens += 1
        embedding_vectors = torch.stack(vectors)
        logger.info(f"generated embedding vector shape: {embedding_vectors.shape}")
        logger.info(f"total number of unknow tokens in pretrained embedding: {total_unknown_tokens}")

    return embedding_vectors
