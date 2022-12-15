from data_process import *
import torch
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Adafactor, BertConfig, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
from models import BayesianBertForSequenceClassification
import logging
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
batch_size = 32
task = 'rte'
pretrained_model_type = 'bert-base-uncased'
torch.cuda.manual_seed(704814)
nlp = load_spacy_nlp(True)
model = EyetrackRegressionBiLSTM(input_size=300, hidden_size=256, linear_output_size=1)
model.load_state_dict(torch.load('savedmodels/nFixPredictionBiLSTM'))
model.eval()
logging.basicConfig(filename=task+'.log', level=logging.DEBUG, format=LOG_FORMAT)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_type)
glue_dataset = datasets.load_dataset('glue', task)
sst2_metric = datasets.load_metric('glue', task)
print("************ Loading Train Dataset ************")
processed_dataset_train = datasets.load_from_disk('glue_dataset/'+task+'_batch32'+'_train')
processed_dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
print('************ Loading Validation Dataset ************')
if task not in ['qnli','qqp','cola','rte','mrpc','sst2','stsb','qqp']:
    processed_dataset_test = glue_dataset['validation'].map(word_spacy_attention_light_version, batched=True, batch_size=batch_size,
                                                        fn_kwargs={'tokenizer': tokenizer, 'task': task,'nlp':nlp,'model':model})
    processed_dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label',
                                                        'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                        'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                        'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
elif task in ['qnli','qqp','cola','rte','mrpc','sst2','stsb','qqp']:
    processed_dataset_valid = datasets.load_from_disk('glue_dataset/'+task+'_batch32'+'_validation')
    processed_dataset_valid.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label',
                                                        'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                        'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                        'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
if task == 'stsb':
    configuration = BertConfig.from_pretrained(pretrained_model_type, num_labels=1)
    config_dict = configuration.to_dict()
    config_dict['scale_func'] = 'tanh'
    config_dict['max_seq_length'] = 128
    configuration=configuration.from_dict(config_dict)
else:
    configuration = BertConfig.from_pretrained(pretrained_model_type)
    config_dict = configuration.to_dict()
    config_dict['scale_func'] = 'tanh'
    config_dict['max_seq_length'] = 128
    configuration=configuration.from_dict(config_dict)
logging.info('************ Loading Model ************')
bayemodel = BayesianBertForSequenceClassification.from_pretrained(pretrained_model_type, config=configuration).cuda()
dataloader_train = torch.utils.data.DataLoader(processed_dataset_train, batch_size=batch_size)
dataloader_valid = torch.utils.data.DataLoader(processed_dataset_valid, batch_size=batch_size)
epoch = 10
param_optimizer = list(bayemodel.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
t_total = len(dataloader_train)* epoch
lr = 3e-5
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
warmup_steps = int(t_total*0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
lam = torch.tensor(1)
step_count = 0
max_length = 128
for _ in range(epoch):
    print('************ Training '+ str(_)+' ************')
    bayemodel.train()
    for batch_idx ,batch in tqdm(enumerate(dataloader_train)):
        step_count +=1
        uid_weight = torch.exp(-step_count*lam)
        input_ids = batch['input_ids'][:,0:max_length].cuda()
        token_type_ids = batch['token_type_ids'][:,0:max_length].cuda()
        attention_mask = batch['attention_mask'][:,0:max_length].cuda()
        labels = batch['label'].cuda()
        attention_matrix_dict = {}
        if task in ['qnli','qqp','cola','rte','mrpc','sst2','stsb','qqp']:
            attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['readability'] = batch['readability'].cuda()
        else:
            attention_matrix_dict['word_length'] = torch.stack(batch['attention_matrix_word_length'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['pos_tag'] = torch.stack(batch['attention_matrix_pos_tag'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['ner'] = torch.stack(batch['attention_matrix_ner'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)[:,0:max_length,0:max_length].cuda()
            attention_matrix_dict['word_position'] = torch.stack(batch['attention_matrix_word_position'], dim=1).cuda()
            attention_matrix_dict['readability'] = batch['readability'].cuda()
        baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions = True, uid_weight = uid_weight)
        loss = baye_outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bayemodel.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if (batch_idx % 5) == 0:
            bayemodel.eval()
            print('************ Evaluation Process ************')
            predict_label = []
            test_label = []
            with torch.no_grad():
                for batch in tqdm(dataloader_valid):
                    input_ids = batch['input_ids'][:,0:max_length].cuda()
                    token_type_ids = batch['token_type_ids'][:,0:max_length].cuda()
                    attention_mask = batch['attention_mask'][:,0:max_length].cuda()
                    labels = batch['label'].cuda()
                    test_label.extend(labels.cpu().tolist())
                    attention_matrix_dict = {}
                    if task in ['qnli','qqp','cola','rte','mrpc','sst2','stsb','qqp']:
                        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'],dim1=-2, dim2=-1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['readability'] = batch['readability'].cuda()
                    else:
                        attention_matrix_dict['word_length'] = torch.stack(batch['attention_matrix_word_length'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['pos_tag'] = torch.stack(batch['attention_matrix_pos_tag'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['ner'] = torch.stack(batch['attention_matrix_ner'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['word_position'] = torch.stack(batch['attention_matrix_word_position'], dim=1)[:,0:max_length,0:max_length].cuda()
                        attention_matrix_dict['readability'] = batch['readability'].cuda()
                    baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                             labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions = True)
                    if task != 'stsb':
                        predict = baye_outputs.logits.argmax(dim=1).tolist()
                    else:
                        predict = baye_outputs.logits.cpu()[:,0].tolist()
                    predict_label.extend(predict)
            results = sst2_metric.compute(predictions=predict_label, references=test_label)
            print('************ Eval results'+' ************')
            if task =='mrpc':
                logging.info('accuracy:'+str(results['accuracy']) +' f1:'+str(results['f1'])+' epoch:'+ str(_) + ' lr = '+str(lr))
            if task =='stsb':
                logging.info('pearson:'+str(results['pearson']) +' spearmanr:'+str(results['spearmanr'])+' epoch:'+ str(_) + ' lr = '+str(lr))
            if task == 'cola':
                logging.info('matthews_correlation:'+str(results['matthews_correlation']) + ' epoch:'+ str(_) + ' lr = '+str(lr))
            if task =='sst2':
                logging.info('accuracy:'+str(results['accuracy']) +' epoch:'+ str(_) + ' lr = '+str(lr))
            if task =='rte':
                logging.info('accuracy:'+str(results['accuracy']) +' epoch:'+ str(_) + ' lr = '+str(lr))
            if task =='qnli':
                logging.info('accuracy:'+str(results['accuracy']) +' epoch:'+ str(_) + ' lr = '+str(lr))
            if task =='qqp':
                logging.info('accuracy:'+str(results['accuracy']) +' f1:'+str(results['f1'])+' epoch:'+ str(_) + ' lr = '+str(lr))
            print(results)
            bayemodel.train()
