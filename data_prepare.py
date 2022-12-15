import torch
import pickle
from transformers import BertTokenizer
from tqdm import tqdm
import tokenizations

class GazeDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, nlp):
        f = open(datapath, 'rb')
        self.data = pickle.load(f)
        self.nlp = nlp
        self.preprocess()
        f.close()

    def preprocess(self):
        for data_idx in tqdm(list(self.data.keys())):
            sent = ' '.join([str(list(self.data[data_idx][x].keys())[0]) for x in list(self.data[data_idx].keys()) if (x != 'sent') & (x != 'inital')])
            doc = self.nlp(sent)
            ori_words = [str(list(self.data[data_idx][x].keys())[0]) for x in list(self.data[data_idx].keys()) if (x != 'sent') & (x != 'inital')]
            feature_matrix, feature_mask = self.feature_assign(doc, ori_words)
            self.data[data_idx]['feature_matrix'] = feature_matrix
            self.data[data_idx]['feature_mask'] = feature_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __getword__(self, sentence_idx, word_idx):
        sentence_data = self.__getitem__(sentence_idx)
        word = list(sentence_data[word_idx].keys())[0]
        return word

    def __getsentencetoken__(self, idx):
        sentence_data = self.__getitem__(idx)
        sentence_words = [list(sentence_data[x].keys())[0] for x in list(sentence_data.keys()) if (x!= 'sent')&(x!= 'inital')]
        return sentence_words

    def feature_assign(self, doc, ori_words):
        feature_matrix = torch.zeros(len(ori_words), 8)
        feature_mask = torch.zeros(len(ori_words), 8)
        spacy_token = [word.text for word in doc]
        a2b, b2a = tokenizations.get_alignments(spacy_token, ori_words)
        track = []
        for word_idx, word in enumerate(doc):
            idx = a2b[word_idx]
            for x in idx:
                if idx in track:
                    feature_matrix[x, 0] = feature_matrix[x, 0]+len(word)
                else:
                    feature_matrix[x, 0] = len(word)
                feature_mask[x, 0] = 1
                feature_matrix[x, 1] = torch.tensor(word_idx/len(doc))
                feature_mask[x, 1] = 1
            if word.ent_type != 0:
                idx = a2b[word_idx]
                for x in idx:
                    feature_matrix[x, 2] = 1
                    feature_mask[x, 2] = 1
            if word.pos_ in ['ADJ', 'ADV', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB', 'INTJ']:
                idx = a2b[word_idx]
                for x in idx:
                    feature_matrix[x, 3] = 1
                    feature_mask[x, 3] = 1
            if word._.assessments != []:
                idx = a2b[word_idx]
                for x in idx:
                    feature_matrix[x, 5] = 1
                    feature_mask[x, 5] = 1
            if word.dep_ in ['det', 'aux', 'cc', 'auxpass', 'dative', 'poss', 'prep']:
                idx = a2b[word_idx]
                for x in idx:
                    feature_matrix[x, 6] = 1
                    feature_mask[x, 6] = 1
            if word.dep_ in ['acomp', 'pcomp', 'pobj']:
                idx = a2b[word_idx]
                for x in idx:
                    feature_matrix[x, 7] = 1
                    feature_mask[x, 7] = 1
            track.append(idx)
        for chunk in doc.noun_chunks:
            for word in chunk:
                idx = a2b[word.i]
                for x in idx:
                    feature_matrix[x, 4] = 1
                    feature_mask[x, 4] = 1
        feature_matrix[:, 0] = feature_matrix[:,0]/torch.sqrt(36+torch.square(feature_matrix[:, 0]))
        return feature_matrix, feature_mask

def gaze_collate_func(batch):
    sent = []
    trt = []
    nFix = []
    feature_matrix = []
    feature_mask = []
    for idx in range(len(batch)):
        sent.append([list(batch[idx][x].keys())[0] for x in list(batch[idx].keys()) if (x != 'sent') & (x != 'inital')&(x != 'feature_matrix')&(x != 'feature_mask')])
        trt.append([batch[idx][x][list(batch[idx][x].keys())[0]]['TRT'] for x in list(batch[idx].keys()) if (x != 'sent') & (x != 'inital')&(x != 'feature_matrix')&(x != 'feature_mask')])
        nFix.append([batch[idx][x][list(batch[idx][x].keys())[0]]['nFix'] for x in list(batch[idx].keys()) if (x != 'sent') & (x != 'inital')&(x != 'feature_matrix')&(x != 'feature_mask')])
        feature_matrix.append(batch[idx]['feature_matrix'])
        feature_mask.append(batch[idx]['feature_mask'])
    return sent, trt, nFix, feature_matrix, feature_mask
