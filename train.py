import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from models import BayesianBertEncoder, BayesianBertModel, BayesianBertForSequenceClassification
from attention_form import *
import spacy
from benepar.spacy_plugin import BeneparComponent
import neuralcoref

nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
batch_size = 8

configuration = BertConfig()
bayemodel = BayesianBertEncoder(configuration)
bayebertmodel = BayesianBertModel(configuration).from_pretrained('bert-base-uncased')
bayebaertseqcls = BayesianBertForSequenceClassification(configuration)

text = 'Hello, my dog is cute'
doc = nlp(text)
attention_matrix = word_length_attention(doc, 1)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = bayebertmodel(input_ids=inputs['input_ids'], token_type_ids=inputs['token_type_ids'],
                        attention_mask=inputs['attention_mask'], attention_matrix=attention_matrix,
                        output_attentions=True)

loss = outputs.loss
logits = outputs.logits