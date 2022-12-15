from data_process import *
import torch
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Adafactor, BertConfig, get_linear_schedule_with_warmup, AutoTokenizer
from matplotlib import pyplot as plt
from models import BayesianBertForTokenClassification
import logging
from tqdm import tqdm
from accelerate import Accelerator


def get_labels(predictions, references):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def compute_metrics(return_entity_level_metrics):
    results = metric.compute()
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
task = 'conll2003'
dataset_type = 'validation'
task_type = 'chunk_tags'
metric = datasets.load_metric('seqeval')
glue_dataset = datasets.load_dataset('conll2003')
batch_size = 32
nlp = load_spacy_nlp(True)
tokenizer, label_to_id, label_list = prepocess_conll(task, dataset_type, task_type)
num_labels = len(label_list)
accelerator = Accelerator()
processed_dataset_train = glue_dataset['validation'].map(tokenize_and_align_labels, batched=True, batch_size=batch_size,
                                                   fn_kwargs={'tokenizer': tokenizer, 'task': 'conll2003',
                                                              'nlp': nlp, 'label_to_id': label_to_id, 'task_type':task_type})
processed_dataset_train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position'])
processed_dataset_test = glue_dataset['test'].map(tokenize_and_align_labels, batched=True, batch_size=batch_size,
                                                   fn_kwargs={'tokenizer': tokenizer, 'task': 'conll2003',
                                                              'nlp': nlp, 'label_to_id': label_to_id, 'task_type':task_type})
processed_dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position'])
configuration = BertConfig(scale_func='tanh', num_labels=num_labels)
bayemodel = BayesianBertForTokenClassification.from_pretrained('bert-base-uncased', config=configuration)
dataloader_train = torch.utils.data.DataLoader(processed_dataset_train, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(processed_dataset_test, batch_size=batch_size)
epoch = 5
param_optimizer = list(bayemodel.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
t_total = len(processed_dataset_train)//batch_size * epoch
warmup_steps = int(t_total*0.1)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
for _ in range(epoch):
   #model.train()
    print('************ Training '+ str(_)+' ************')
    bayemodel.train()
    for batch_idx, batch in tqdm(enumerate(dataloader_train)):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        attention_matrix_dict = {}
        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'],dim1=-2, dim2=-1)
        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'],dim1=-2, dim2=-1)
        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'],dim1=-2, dim2=-1)
        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'],dim1=-2, dim2=-1)
        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)
        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)
        #outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
        baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions=True)
        loss = baye_outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bayemodel.parameters(), 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if (batch_idx % 50) == 0:
            bayemodel.eval()
            print('************ Evaluation Process ************')
            for batch in tqdm(dataloader_test):
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                attention_matrix_dict = {}
                attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2,
                                                                        dim2=-1)
                attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
                attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
                attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'],
                                                                          dim1=-2, dim2=-1)
                attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)
                attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)
                baye_outputs = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                         labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions = True)

                predictions = baye_outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)
                preds, refs = get_labels(predictions_gathered, labels_gathered)
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                ) # predictions and preferences are expected to be a nested list of labels, not label_ids
            eval_metric = compute_metrics(False)
            print('************ Eval results'+' ************')
            bayemodel.train()