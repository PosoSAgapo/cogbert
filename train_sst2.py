from data_process import *
import torch
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Adafactor, BertConfig, get_linear_schedule_with_warmup
from models import BayesianBertForSequenceClassification, EyetrackRegressionBiLSTM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
glue_dataset = datasets.load_dataset('glue', 'sst2')
sst2_metric = datasets.load_metric('glue', 'sst2')
batch_size = 8
task = 'cola'
nlp = load_spacy_nlp(True)
model = EyetrackRegressionBiLSTM(input_size=300, hidden_size=256, linear_output_size=1)
model.load_state_dict(torch.load('nFixPredictionBiLSTM'))
model.eval()
processed_dataset_train = datasets.load_from_disk('glue_dataset/'+task+'_batch32'+'_train')
processed_dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
# processed_dataset_test = glue_dataset['validation'].map(word_spacy_attention, batched=True, batch_size=batch_size,
#                                                         fn_kwargs={'tokenizer': tokenizer, 'task': task,'nlp':nlp})
processed_dataset_test = datasets.load_from_disk('glue_dataset/'+task+'_batch32'+'_validation')
processed_dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
configuration = BertConfig(scale_func='tanh')
bayemodel = BayesianBertForSequenceClassification.from_pretrained('bert-base-uncased', config=configuration)
dataloader_train = torch.utils.data.DataLoader(processed_dataset_train, batch_size=8)
dataloader_test = torch.utils.data.DataLoader(processed_dataset_test, batch_size=8)
epoch = 5
param_optimizer = list(bayemodel.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
warmup_steps = 0
t_total = len(processed_dataset_test)//batch_size * epoch
warmup_steps = int(t_total*0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
loss_list = []
for _ in range(epoch):
   #model.train()
    bayemodel.train()
    for batch in dataloader_train:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        attention_matrix_dict = {}
        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2,dim2=-1)
        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'], dim1=-2,dim2=-1)
        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)
        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)
        attention_matrix_dict['readability'] = batch['readability']
        baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions=True)
        loss = baye_outputs.loss
        loss_list.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    bayemodel.eval()
    predict_label = []
    test_label = []
    for batch in dataloader_test:
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        test_label.extend(labels)
        attention_matrix_dict = {}
        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2, dim2=-1)
        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'], dim1=-2,
                                                                  dim2=-1)
        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)
        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)
        attention_matrix_dict['readability'] = batch['readability']
        baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels, attention_matrix_dict=attention_matrix_dict)
        predict = baye_outputs.logits.argmax().cpu().tolist()
        predict_label.extend(predict)
    results = sst2_metric.compute(predictions=predict_label, reference=test_label)
    print(results)
