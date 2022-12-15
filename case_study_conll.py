import torch
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification, AdamW, Adafactor, BertConfig, get_linear_schedule_with_warmup
from models import BayesianBertForSequenceClassification, BayesianBertForTokenClassification, EyetrackRegressionBiLSTM
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
from data_process import *
import pickle

def get_labels(predictions, references):
    # Transform predictions and references tensos to numpy arrays
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

def return_sentence(example):
    print(example['tokens'])
    return example

def collate_fn(batch):
    sentence = []
    for data in batch:
        sentence.append(' '.join(data['tokens']))
    return sentence

metric = datasets.load_metric('seqeval')
glue_dataset = datasets.load_dataset('conll2003')
test_dataset = glue_dataset['test'].map(return_sentence)
dataloader_sentence = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

#label_to_id, label_list = prepocess_conll(glue_dataset, dataset_type, task_type, tokenizer)

processed_dataset_test = datasets.load_from_disk('/Users/chenbowen/Documents/PaperCode/BayesianAttentionBert/glue_dataset/conll2003_batch32_test_ner_tags_cased')
processed_dataset_test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels',
                                                    'attention_matrix_pos_tag', 'attention_matrix_ner',
                                                    'attention_matrix_emotion', 'attention_matrix_npchunk',
                                                    'attention_matrix_word_length', 'attention_matrix_word_position','readability'])
model_type = 'bert-base-cased'
configuration = BertConfig.from_pretrained(model_type, num_labels=9)
config_dict =configuration.to_dict()
config_dict['scale_func'] = 'tanh'
configuration=configuration.from_dict(config_dict)
tokenizer = BertTokenizer.from_pretrained(model_type)
batch_size = 16
bayemodel = BayesianBertForTokenClassification.from_pretrained(model_type, config=configuration)
bayemodel.load_state_dict(torch.load('savedmodels/ner_cog_bert', map_location=torch.device('cpu')))
bayemodel = bayemodel.eval()
model = BertForTokenClassification.from_pretrained(model_type, config=configuration)
model.load_state_dict(torch.load('savedmodels/ner_bert', map_location=torch.device('cpu')))
model.eval()
dataloader_test = torch.utils.data.DataLoader(processed_dataset_test, batch_size=batch_size)
with torch.no_grad():
    for batch, batch_sentence in zip(tqdm(dataloader_test), (dataloader_sentence)):
        input_ids = batch['input_ids']
        token_list = []
        for i in range(input_ids.shape[0]):
            token_list.append(tokenizer.convert_ids_to_tokens(input_ids[i]))
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        attention_matrix_dict = {}
        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2,dim2=-1)
        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'],dim1=-2, dim2=-1)
        attention_matrix_dict['emotion'] = torch.stack(batch['attention_matrix_emotion'], dim=1)
        attention_matrix_dict['npchunk'] = torch.stack(batch['attention_matrix_npchunk'], dim=1)
        attention_matrix_dict['readability'] = batch['readability']
        baye_outputs, attention_weight = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions=True)
        model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,
                              output_attentions=True)
        predictions = baye_outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        baye_attentions = baye_outputs.attentions
        model_attentions = model_outputs.attentions
        1+'p'+[]
        sns.set()
        for data_idx in range(input_ids.shape[0]):
            sentence = batch_sentence[data_idx]
            print(sentence)
            data_idx = 3
        for head_idx in range(12):
            layer_idx = 11
            head_idx = head_idx
            non_padded_length = len(input_ids[data_idx][input_ids[data_idx] != 0])
            # fig, (ax, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios':[1, 1]})
            fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 7))
            xLabel = token_list[data_idx][0:non_padded_length]
            yLabel = token_list[data_idx][0:non_padded_length]
            for i in range(len(xLabel)):
                xLabel[i] = xLabel[i].replace('#', '')
                yLabel[i] = yLabel[i].replace('#', '')
            baye_layer_head_i_attention = baye_attentions[layer_idx][data_idx, head_idx, 0:non_padded_length,
                                          0:non_padded_length]
            model_layer_head_i_attention = model_attentions[layer_idx][data_idx, head_idx, 0:non_padded_length, 0:non_padded_length]
            vmin = 0
            baye_max = torch.max(baye_layer_head_i_attention)
            model_max = torch.max(model_layer_head_i_attention)
            if baye_max > model_max:
                max = baye_max
            else:
                max = model_max

            sns.heatmap(baye_layer_head_i_attention.data, ax=ax,
                        cmap=sns.light_palette("#2ecc71", as_cmap=True),
                        xticklabels=xLabel, yticklabels=yLabel, cbar=False, square=True, vmin=vmin, vmax=max)
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(left=False, top=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
            ax.set_yticklabels(ax.get_xticklabels(), fontsize=10)
            ax.set_title('Our model', fontsize=12)
            ax2 = sns.heatmap(model_layer_head_i_attention.data, ax=ax2,
                              cmap=sns.light_palette("#2ecc71", as_cmap=True),
                              xticklabels=xLabel, yticklabels=yLabel, cbar=False, square=True, vmin=vmin, vmax=max)
            ax2.xaxis.set_ticks_position('top')
            ax2.tick_params(left=False, top=False)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=10)
            ax2.set_yticklabels(ax2.get_xticklabels(), fontsize=10)
            ax2.set_title('Bert model', fontsize=12)
            cbar_ax = fig.add_axes([0.658, 0.24, 0.01, 0.55])
            # fig.colorbar(ax2.collections[0], cax=cbar_ax, shrink=0.5)
            fig.colorbar(ax2.collections[0], cax=cbar_ax)

            # fig.tight_layout()
            # plt.text(-50, -10, sentence)
            plt.show()
save_dict = {'tokenized_sentence': xLabel, 'bert_attention': model_attentions, 'baye_attention': baye_attentions}
f = open('conll2003_case.pkl', 'wb')
pickle.dump(save_dict, f)
f.close()
#first batch
#layer id 11 head_idx 6 has a example of fun being attended
#layer id 11 head_idx 11 has a example of meaningful words being attended
#layer id 11 head_idx 3 has a example of sentence being attended
#layer id 11 head_idx 4 has a example of fun example being attended
#layer id 11 head_idx 5 has a example
#layer id 3 head_idx 0 has a example