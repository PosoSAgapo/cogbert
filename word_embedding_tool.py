from torchtext.vocab import Vectors
import torch
from nltk.corpus import wordnet as wn


vectors = Vectors('glove.840B.300d.txt')



def remove_special_tokens(token):
    token = token.replace(',', '').replace(';', '').replace(':', '').replace('.', '').replace('!', '').replace('?','').\
        replace('......', '').replace('"','')\
        if token == str else str(token)
    return token


def get_embedding(sentence_batch, dimension=300):
    length = [len(x) for x in sentence_batch]
    max_length = max(length)
    embedding = torch.zeros(max_length, len(sentence_batch), dimension)
    for data, data_idx in zip(sentence_batch, range(len(sentence_batch))):
        if data != []:
            for token, idx in zip(data, range(len(data))):
                embedding[idx, data_idx, :] = vectors[remove_special_tokens(token)]
        else:
            length[data_idx] = 1
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(embedding, length, enforce_sorted=False)
    return packed_sequence

def get_target(eyetrack_data, dimension=1):
    length = [len(x) for x in eyetrack_data]
    max_length = max(length)
    embedding = torch.zeros(max_length, len(eyetrack_data), 1)
    for data, data_idx in zip(eyetrack_data, range(len(eyetrack_data))):
        for token, idx in zip(data, range(len(data))):
            embedding[idx, data_idx, :] = data[idx]
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(embedding, length, enforce_sorted=False)
    return packed_sequence

def get_feature_mask(feature, mask):
    length = [len(x) for x in feature]
    max_length = max(length)
    embedding = torch.zeros(max_length, len(feature), 8)
    erase = torch.zeros(max_length, len(mask), 8)
    for data, sub_mask, data_idx in zip(feature, mask, range(len(feature))):
        for token, idx in zip(data, range(len(data))):
            embedding[idx, data_idx, :] = data[idx]
            erase[idx, data_idx, :] = sub_mask[idx]
    packed_feature = torch.nn.utils.rnn.pack_padded_sequence(embedding, length, enforce_sorted=False)
    packed_mask = torch.nn.utils.rnn.pack_padded_sequence(erase, length, enforce_sorted=False)
    return packed_feature, packed_mask