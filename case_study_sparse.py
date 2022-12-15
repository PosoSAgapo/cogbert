import torch
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Adafactor, BertConfig, get_linear_schedule_with_warmup
from models import BayesianBertForSequenceClassification, BayesianBertForTokenClassification, EyetrackRegressionBiLSTM
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import spacy
import pickle

def return_sentence(example):
    return example

def collate_fn(batch):
    np_chunk = []
    emotion = []
    syntax_dac = []
    syntax_pcob = []
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    ner_list = []
    pos_list = []
    word_postion = []
    word_length = []
    readability_list = []
    for i in range(len(batch)):
        input_ids = batch[i]['input_ids']
        token_type_ids = batch[i]['token_type_ids']
        attention_mask = batch[i]['attention_mask']

        emotion_indice = batch[i]['emotion_indice']
        npchunk_indice = batch[i]['npchunk_indice']
        syntax_dac_indice = batch[i]['syntax_dac_indice']
        syntax_pcob_indice = batch[i]['syntax_pcob_indice']

        emotion_value = batch[i]['emotion_value']
        npchunk_value = batch[i]['npchunk_value']
        syntax_dac_value = batch[i]['syntax_dac_value']
        syntax_pcob_value = batch[i]['syntax_pcob_value']

        emotion_size = batch[i]['emotion_size']
        npchunk_size = batch[i]['npchunk_size']
        syntax_dac_size = batch[i]['syntax_dac_size']
        syntax_pcob_size = batch[i]['syntax_pcob_size']

        label_list.append(batch[i]['label'])
        readability_list.append(batch[i]['readability'])
        ner_list.append(batch[i]['attention_matrix_ner'])
        pos_list.append(batch[i]['attention_matrix_pos_tag'])
        word_postion.append(batch[i]['attention_matrix_word_position'])
        word_length.append(batch[i]['attention_matrix_word_length'])
        np_chunk_matrix = torch.sparse_coo_tensor(npchunk_indice, npchunk_value, npchunk_size)
        emotion_matrix = torch.sparse_coo_tensor(emotion_indice, emotion_value, emotion_size)
        syntax_dac_matrix = torch.sparse_coo_tensor(syntax_dac_indice, syntax_dac_value, syntax_dac_size)
        syntax_pcob_matrix = torch.sparse_coo_tensor(syntax_pcob_indice, syntax_pcob_value, syntax_pcob_size)

        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        np_chunk.append(np_chunk_matrix.to_dense())
        emotion.append(emotion_matrix.to_dense())
        syntax_dac.append(syntax_dac_matrix.to_dense())
        syntax_pcob.append(syntax_pcob_matrix.to_dense())

    input_ids = torch.tensor(input_ids_list)
    token_type_ids = torch.tensor(token_type_ids_list)
    attention_mask = torch.tensor(attention_mask_list)
    label = torch.tensor(label_list)
    readability = torch.tensor(readability_list)
    ner_matrix = torch.tensor(ner_list)
    pos_matrix = torch.tensor(pos_list)
    word_length_matrix = torch.tensor(word_length)
    word_position_matrix = torch.tensor(word_postion)
    np_chunk = torch.stack(np_chunk, dim=0)
    emotion_matrix = torch.stack(emotion, dim=0)
    syntax_dac_matrix = torch.stack(syntax_dac, dim=0)
    syntax_pcob_matrix = torch.stack(syntax_pcob, dim=0)
    batch = {}
    batch['input_ids'] = input_ids
    batch['token_type_ids'] = token_type_ids
    batch['attention_mask'] = attention_mask
    batch['attention_matrix_emotion'] = emotion_matrix
    batch['attention_matrix_npchunk'] = np_chunk
    batch['attention_matrix_syntax_dac'] = syntax_dac_matrix
    batch['attention_matrix_syntax_pcob'] = syntax_pcob_matrix
    batch['label'] = label
    batch['readability'] = readability
    batch['attention_matrix_ner'] = ner_matrix
    batch['attention_matrix_pos_tag'] = pos_matrix
    batch['attention_matrix_word_length'] = word_length_matrix
    batch['attention_matrix_word_position'] = word_position_matrix
    return batch


task = 'mrpc'
batch_size = 32
pretrained_model_type = 'bert-base-uncased'
processed_dataset = datasets.load_from_disk('/Users/chenbowen/Documents/PaperCode/BayesianAttentionBert/glue_dataset_sparse/'+task+'_batch32_sparse')
processed_dataset.set_format(type=None, columns=['input_ids','token_type_ids', 'attention_mask', 'label', 'attention_matrix_ner',
                                                 'attention_matrix_pos_tag', 'attention_matrix_word_length','attention_matrix_word_position',
                                                 'emotion_indice', 'emotion_value', 'emotion_size',
                                                 'npchunk_indice','npchunk_value', 'npchunk_size',
                                                 'syntax_dac_indice', 'syntax_dac_value', 'syntax_dac_size',
                                                 'syntax_pcob_indice', 'syntax_pcob_value', 'syntax_pcob_size', 'readability'])
max_length = 128
glue_dataset = datasets.load_dataset('glue', task)
nlp = spacy.load('en_core_web_md')
validaiton_dataset = glue_dataset['validation'].map(return_sentence, batch_size=32)
if task == 'stsb':
    configuration = BertConfig.from_pretrained(pretrained_model_type, num_labels=1)
    baye_config_dict = configuration.to_dict()
    baye_config_dict['scale_func'] = 'tanh'
    baye_config_dict['max_seq_length'] = max_length
    baye_configuration=configuration.from_dict(baye_config_dict)
    normal_config_dict = configuration.to_dict()
    normal_config_dict['max_seq_length'] = max_length
    normal_config_dict = configuration.from_dict(normal_config_dict)
elif task == 'mnli':
    configuration = BertConfig.from_pretrained(pretrained_model_type, num_labels=3)
    baye_config_dict = configuration.to_dict()
    baye_config_dict['scale_func'] = 'tanh'
    baye_config_dict['max_seq_length'] = max_length
    baye_configuration = configuration.from_dict(baye_config_dict)
    normal_config_dict = configuration.to_dict()
    normal_config_dict['max_seq_length'] = max_length
    normal_config_dict = configuration.from_dict(normal_config_dict)
else:
    configuration = BertConfig.from_pretrained(pretrained_model_type)
    baye_config_dict = configuration.to_dict()
    baye_config_dict['scale_func'] = 'tanh'
    baye_config_dict['max_seq_length'] = max_length
    baye_configuration = configuration.from_dict(baye_config_dict)
    normal_config_dict = configuration.to_dict()
    normal_config_dict['max_seq_length'] = max_length
    normal_config_dict = configuration.from_dict(normal_config_dict)
bayemodel = BayesianBertForSequenceClassification.from_pretrained(pretrained_model_type, config=baye_configuration)
bayemodel.load_state_dict(torch.load('saved_sparse_model/'+task+'_cog_bert', map_location=torch.device('cpu')))
bayemodel.eval()
model = BertForSequenceClassification.from_pretrained(pretrained_model_type, config=normal_config_dict)
model.load_state_dict(torch.load('savedmodels/'+task+'_bert', map_location=torch.device('cpu')))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("************ Spliting Train Dataset ************")
dataloader_train = torch.utils.data.DataLoader(processed_dataset['train'], batch_size=batch_size, collate_fn=collate_fn)
print("************ Spliting Valid Dataset ************")
dataloader_valid = torch.utils.data.DataLoader(processed_dataset['validation'], batch_size=batch_size, collate_fn=collate_fn)
dataloader_sentence = torch.utils.data.DataLoader(validaiton_dataset, batch_size=batch_size)

print("************ Spliting Test Dataset ************")
dataloader_test = torch.utils.data.DataLoader(processed_dataset['test'], batch_size=batch_size, collate_fn=collate_fn)
mean_feature_weight = torch.zeros(8)
scale_func = torch.nn.Softmax()
batch_idx = 0
with torch.no_grad():
    for batch, batch_sentence in zip(tqdm(dataloader_valid), (dataloader_sentence)):
        input_ids = batch['input_ids']
        token_list = []
        for i in range(input_ids.shape[0]):
            token_list.append(tokenizer.convert_ids_to_tokens(input_ids[i]))
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        attention_matrix_dict = {}
        total_label_lower = 0
        total_label_higher = 0
        total_label = 0
        data_ratio = torch.ones(8)
        attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2,
                                                                dim2=-1)
        attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
        attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
        attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'], dim1=-2,
                                                                  dim2=-1)
        attention_matrix_dict['emotion'] = batch['attention_matrix_emotion']
        attention_matrix_dict['npchunk'] = batch['attention_matrix_npchunk']
        attention_matrix_dict['syntax_dac'] = batch['attention_matrix_syntax_dac']
        attention_matrix_dict['syntax_pcob'] = batch['attention_matrix_syntax_pcob']
        attention_matrix_dict['readability'] = batch['readability']
        baye_outputs, attention_weight = bayemodel(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 labels=labels, attention_matrix_dict=attention_matrix_dict, output_attentions=True)
        model_outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, output_attentions=True)
        baye_attentions = baye_outputs.attentions
        model_attentions = model_outputs.attentions
        for key_idx, key in enumerate(list(attention_matrix_dict.keys())[0:8]):
            if key_idx <= 3:
                total_label = total_label + torch.sum(attention_matrix_dict[key] != 0)
            else:
                total_label = total_label + torch.sum(attention_matrix_dict[key] != 0)
        for key_idx, key in enumerate(list(attention_matrix_dict.keys())[0:8]):
            if key_idx <= 3:
                data_ratio[key_idx] = torch.sum(attention_matrix_dict[key] != 0) / total_label
            else:
                data_ratio[key_idx] = torch.sum(attention_matrix_dict[key] != 0) / total_label
            if data_ratio[key_idx] < 0.1:
                data_ratio[key_idx] = 0.14
        print(data_ratio)
        data_ratio = data_ratio / torch.max(data_ratio)
        batch_mean_weight = attention_weight.mean(dim=0)
        batch_mean_weight[0:3] = batch_mean_weight[0:3] * 1 / 3 * (1 / data_ratio[0:3])
        batch_mean_weight[4:] = batch_mean_weight[4:] * 1/ 3 * (1 / data_ratio[4:])
        mean_feature_weight = mean_feature_weight + batch_mean_weight
        sns.set()
        batch_idx = batch_idx + 1
        if batch_idx == 1 :
            input()
        for data_idx in range(input_ids.shape[0]):
            data_idx = 11
            if task in ['mrpc','rte']:
                sentence1 = batch_sentence['sentence1'][data_idx]
                sentence2 = batch_sentence['sentence2'][data_idx]
                #print('sentence1: '+sentence1+'sentence2: '+sentence2)
                doc = nlp(sentence1+sentence2)
                for word in doc:
                    #if word.dep_ in ['acomp', 'pcomp', 'pobj']:
                    if word.dep_ in ['aux', 'cc', 'poss', 'prep']:
                    #if word.dep_ in ['aux', 'cc', 'poss', 'prep','acomp', 'pcomp', 'pobj']:
                        print(
                            'word: ' + word.text + ' dep: ' + word.dep_ + ' head: ' + word.head.text + ' sentence: ' + doc.text)

            else:
                sentence = batch_sentence['sentence'][data_idx]
                doc = nlp(sentence)
                for word in doc:
                    #if word.dep_ in ['acomp', 'pcomp', 'pobj']:
                    if word.dep_ in ['aux', 'cc', 'poss', 'prep']:
                    #if word.dep_ in ['aux', 'cc', 'poss', 'prep', 'acomp', 'pcomp', 'pobj']:
                        print(
                            'word: ' + word.text + ' dep: ' + word.dep_ + ' head: ' + word.head.text + ' sentence: ' + doc.text)

                #print(sentence)
        #for head_idx in range(0, 12):
            layer_idx = 9
            head_idx = 0
            non_padded_length = len(input_ids[data_idx][input_ids[data_idx] != 0])
            # fig, (ax, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios':[1, 1]})
            fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 7), gridspec_kw={'width_ratios': [4, 4, 1]})
            xLabel = token_list[data_idx][0:non_padded_length]
            yLabel = token_list[data_idx][0:non_padded_length]
            for i in range(len(xLabel)):
                xLabel[i] = xLabel[i].replace('#', '')
                yLabel[i] = yLabel[i].replace('#', '')
            baye_layer_head_i_attention = baye_attentions[layer_idx][data_idx, head_idx, 0:non_padded_length,
                                          0:non_padded_length]
            model_layer_head_i_attention = model_attentions[layer_idx][data_idx, head_idx, 0:non_padded_length,
                                           0:non_padded_length]
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
            cbar_ax = fig.add_axes([0.828, 0.105, 0.01, 0.725])
            # fig.colorbar(ax2.collections[0], cax=cbar_ax, shrink=0.5)
            fig.colorbar(ax2.collections[0], cax=cbar_ax)

            # fig.tight_layout()
            # plt.text(-50, -10, sentence)
            plt.show()

save_dict = {'tokenized_sentence': xLabel, 'bert_attention': model_attentions, 'baye_attention': baye_attentions, 'data_idx': 11}
f = open('new_mrpc_case.pkl', 'wb')
pickle.dump(save_dict, f)
f.close()
