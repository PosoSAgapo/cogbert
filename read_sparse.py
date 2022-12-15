import datasets
import torch


def collate_fn(batch):
    print(len(batch))
    np_chunk = []
    emotion = []
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
        emotion_value = batch[i]['emotion_value']
        npchunk_value = batch[i]['npchunk_value']
        emotion_size = batch[i]['emotion_size']
        npchunk_size = batch[i]['npchunk_size']
        label_list.append(batch[i]['label'])
        readability_list.append(batch[i]['readability'])
        ner_list.append(batch[i]['attention_matrix_ner'])
        pos_list.append(batch[i]['attention_matrix_pos_tag'])
        word_postion.append(batch[i]['attention_matrix_word_position'])
        word_length.append(batch[i]['attention_matrix_word_length'])
        np_chunk_matrix = torch.sparse_coo_tensor(npchunk_indice, npchunk_value, npchunk_size)
        emotion_matrix = torch.sparse_coo_tensor(emotion_indice, emotion_value, emotion_size)
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        np_chunk.append(np_chunk_matrix.to_dense())
        emotion.append(emotion_matrix.to_dense())
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
    batch = {}
    batch['input_ids'] = input_ids
    batch['token_type_ids'] = token_type_ids
    batch['attention_mask'] = attention_mask
    batch['attention_matrix_emotion'] = emotion_matrix
    batch['attention_matrix_npchunk'] = np_chunk
    batch['label'] = label
    batch['readability'] = readability
    batch['attention_matrix_ner'] = ner_matrix
    batch['attention_matrix_pos_tag'] = pos_matrix
    batch['attention_matrix_word_length'] = word_length_matrix
    batch['attention_matrix_word_position'] = word_position_matrix
    return batch

processed_dataset_train = datasets.load_from_disk('/Users/chenbowen/Documents/PaperCode/BayesianAttentionBert/glue_dataset/mnli_batch32_train_sparse')
processed_dataset_train.set_format(type=None, columns=['input_ids','token_type_ids','attention_mask','label', 'attention_matrix_ner', 'attention_matrix_pos_tag', 'attention_matrix_word_length',
                                                       'attention_matrix_word_position', 'emotion_indice', 'npchunk_indice',
                                                       'emotion_value','npchunk_value','emotion_size','npchunk_size','readability'])
dataloader_train = torch.utils.data.DataLoader(processed_dataset_train, batch_size=32, collate_fn = collate_fn)

for idx, batch in enumerate(dataloader_train):
    input_ids = batch['input_ids']
    token_type_ids = batch['token_type_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']
    attention_matrix_dict = {}
    attention_matrix_dict['word_length'] = torch.diag_embed(batch['attention_matrix_word_length'], dim1=-2, dim2=-1)
    attention_matrix_dict['pos_tag'] = torch.diag_embed(batch['attention_matrix_pos_tag'], dim1=-2, dim2=-1)
    attention_matrix_dict['ner'] = torch.diag_embed(batch['attention_matrix_ner'], dim1=-2, dim2=-1)
    attention_matrix_dict['word_position'] = torch.diag_embed(batch['attention_matrix_word_position'], dim1=-2,dim2=-1)
    attention_matrix_dict['emotion'] = batch['attention_matrix_emotion']
    attention_matrix_dict['npchunk'] = batch['attention_matrix_npchunk']
    attention_matrix_dict['readability'] = batch['readability']