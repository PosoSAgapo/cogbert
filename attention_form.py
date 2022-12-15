import spacy
import pickle
from benepar.spacy_plugin import BeneparComponent
from nltk.tree import Tree
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import neuralcoref


nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')


def dependecy_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512, batchsize))
    for word, word_idx in zip(doc, range(len(doc))):
        for child_word in [(child.text, child.i) for child in word.children]:
            child_word_idx = child_word[1]
            attention_matrix[word_idx][child_word_idx] = 1
    return dependecy_attention

def traverse_tree_spacy(tree, batchsize):
    score_dict = {}
    attention_matrix = torch.zeros((512, 512, batchsize))
    for word, word_idx in zip(tree, range(len(tree))):
        score_dict[word_idx] = {}
        score_dict[word_idx]['text'] = word
        score_dict[word_idx]['score'] = 0
        score_dict[word_idx]['connected'] = []
    for consitituent in tree._.constituents:
        #print(consitituent._.parse_string)
        chunk_type = consitituent._.parse_string.split()[0].replace('(', '')
        if (chunk_type == 'NP') | (chunk_type == 'VP') | (chunk_type == 'PP'):
            for word, word_idx in zip(consitituent, range(len(consitituent))):
                score_dict[word.i]['score'] = score_dict[word.i]['score'] + 1
                word_list = list(consitituent)
                word_list.pop(word_idx)
                score_dict[word.i]['connected'].append(word_list)
        elif not chunk_type.isalpha():
            for word, word_idx in zip(consitituent, range(len(consitituent))):
                score_dict[word.i]['score'] = score_dict[word.i]['score'] + 0.5
        elif chunk_type.isalpha():
            pass
    for word_idx in score_dict.keys():
        attention_matrix[word_idx][word_idx] = score_dict[word_idx]['score']
        for consitituent in score_dict[word_idx]['connected']:
           for consitituent_word in consitituent:
               attention_matrix[word_idx][consitituent_word.i] = attention_matrix[word_idx][consitituent_word.i] + 1
    return attention_matrix


def coreference_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512,batchsize))
    for coreference_cluster in doc._.coref_clusters:
        for word_span in coreference_cluster.mentions:
            for word in word_span:
                for target_word_span in coreference_cluster.mentions:
                    for target_word in target_word_span:
                        attention_matrix[word.i][target_word.i] = attention_matrix[word.i][target_word.i] + 1
    return attention_matrix


def seperator_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512, batchsize))
    seperator = []
    for word in doc:
        if (word.text == ',')|(word.text == ':'):
            if not seperator:
                attention_matrix[0:word.i, 0:word.i] = attention_matrix[0:word.i, 0:word.i] + 1
                seperator.append(word.i)
            else:
                attention_matrix[seperator[-1]:word.i, seperator[-1]:word.i] = attention_matrix[seperator[-1]:word.i, seperator[-1]:word.i] + 1
    return attention_matrix

def word_length_attention(doc, batchsize):
    attention_matrix = torch.zeros((batchsize, 512, 512))
    for data_idx in range(batchsize):
        for word in doc:
            attention_matrix[data_idx, word.i, word.i] = len(word.text)
    return attention_matrix

def content_word_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512, batchsize))
    for word in doc:
        if word.pos_ in ['ADJ', 'ADV', 'NOUN', 'NUM', 'PRON', 'PROPN', 'VERB', 'INTJ']:
            attention_matrix[word.i][word.i] = attention_matrix[word.i][word.i] + 5
        else:
            attention_matrix[word.i][word.i] = attention_matrix[word.i][word.i] + 1
    return attention_matrix

def ent_word_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512, batchsize))
    for ent in doc.ents:
        for word in ent:
            for target_word in ent:
                attention_matrix[word.i][target_word.i] = attention_matrix[word.i][target_word.i] + 1
    return attention_matrix

def emotion_word_attention(doc, batchsize):
    attention_matrix = torch.zeros((512, 512, batchsize))
    for word in doc:
        pass
    return attention_matrix


# sample_text = 'Henry Ford, with his son Edsel, founded the Ford Foundation in 1936 as a local philanthropic organization with a broad charter to promote human welfare.'
# doc = nlp(sample_text)
# attention_matrix = torch.zeros((512, 512,batchsize))
# word_list = [word.text for word in doc]
# for word, word_idx in zip(doc, range(len(doc))):
#     print([child.i for child in word.children])
#     print(word.text, word.head.text, [child for child in word.children])
#     for child_word in [(child.text, child.i) for child in word.children]:
#         child_word_text = child_word[0]
#         child_word_idx = child_word[1]
#         attention_matrix[word_idx][child_word_idx] = 1
# np_chunked_sentence = list(doc.sents)[0]._.parse_string
# np_tree = Tree.fromstring(np_chunked_sentence._.parse_string)
#
# sns.set()
# xLabel = word_list
# yLabel = word_list
# ax = sns.heatmap(attention_matrix.data[ :len(word_list), :len(word_list)],
#                  cmap=sns.light_palette("#2ecc71", as_cmap=True),
#                  xticklabels=word_list, yticklabels=word_list)
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(left=False, top=False)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.show()




