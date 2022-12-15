from models import *
from data_prepare import *
from word_embedding_tool import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from benepar.spacy_plugin import BeneparComponent
from functools import partial
import seaborn as sns
import numpy as np

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

random_seed = 43
dataset = GazeDataset(datapath='eyetrack_data/average_sentences.pkl', nlp=nlp)
train_size, validate_size = int(0.8*len(dataset))+1, int(0.2*len(dataset))
train_set, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size], generator=torch.Generator().manual_seed(random_seed))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, collate_fn=gaze_collate_func)
eval_loader = torch.utils.data.DataLoader(validate_set, batch_size=validate_size, shuffle=False, collate_fn=gaze_collate_func)
model = EyetrackRegressionBiLSTM(input_size=300, hidden_size=256, linear_output_size=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
epoch = 40
mse_loss_func = torch.nn.MSELoss()
feature_loss = []
target_loss = []
mse_loss = []
for _ in tqdm(range(epoch)):
    for data_idx, train_batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        if data_idx == 40:
            a = 1+1
            pass
        words = train_batch[0]
        trt = train_batch[1]
        nFix = train_batch[2]
        target_feature = train_batch[3]
        feature_mask = train_batch[4]
        train_embedding = get_embedding(words)
        target = get_target(nFix)
        packed_feature, packed_mask = get_feature_mask(target_feature, feature_mask)
        embedding, embedding_length = torch.nn.utils.rnn.pad_packed_sequence(train_embedding)
        predcited_nFix, predicted_feature, unpacked_sentence_length = model(train_embedding, packed_mask, packed_feature, data_idx)
        target_unpacked_output, target_unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(target)
        feature_unpacked_output, feature_unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(packed_feature)
        feature_target_loss = mse_loss_func(predicted_feature, feature_unpacked_output[:, :, 2:])
        nFix_target_loss = mse_loss_func(predcited_nFix, target_unpacked_output)
        final_loss = feature_target_loss + nFix_target_loss
        final_loss.backward()
        optimizer.step()
    model.eval()
    for eval_data_idx, eval_batch in tqdm(enumerate(eval_loader)):
        words = eval_batch[0]
        trt = eval_batch[1]
        nFix = eval_batch[2]
        target_feature = eval_batch[3]
        feature_mask = eval_batch[4]
        eval_embedding = get_embedding(words)
        target = get_target(nFix)
        packed_feature, packed_mask = get_feature_mask(target_feature, feature_mask)
        predcited_nFix, predicted_feature, unpacked_sentence_length = model(eval_embedding, packed_mask, packed_feature, eval_data_idx)
        target_unpacked_output, target_unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(target)
        feature_unpacked_output, feature_unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(packed_feature)
        feature_target_loss = mse_loss_func(predicted_feature, feature_unpacked_output[:, :, 2:])
        nFix_target_loss = mse_loss_func(predcited_nFix, target_unpacked_output)
        final_loss = feature_target_loss + nFix_target_loss
        mse_loss.append(final_loss.item())
        target_loss.append(nFix_target_loss.item())
        feature_loss.append(feature_target_loss.item())
    model.train()
plt.plot(mse_loss, 'y', label='MSE Loss')
plt.plot(target_loss, 'r', label='Target Loss')
plt.plot(feature_loss, 'b', label='Feature Loss')
plt.show()
#torch.save(model.state_dict(), 'savedmodels/nFixPredictionBiLSTM')

data = dataset.data
select_data = data[5]
words = [list(select_data[x].keys())[0] for x in list(select_data.keys()) if (x != 'sent') & (x != 'inital')&(x != 'feature_matrix')&(x != 'feature_mask')]
nFix = [select_data[x][list(select_data[x].keys())[0]]['nFix'] for x in list(select_data.keys()) if (x != 'sent') & (x != 'inital')&(x != 'feature_matrix')&(x != 'feature_mask')]
feature = select_data['feature_matrix']
mask = select_data['feature_mask']

eval_embedding = get_embedding([words])
target = get_target([nFix])
packed_feature, packed_mask = get_feature_mask([feature], [mask])

model.eval()
predcited_nFix, predicted_feature, unpacked_sentence_length = model(eval_embedding, packed_mask, packed_feature, eval_data_idx)
xLabel = words
proj_x = range(len(xLabel))
feature_name = ['Word Length', 'Word Position', 'Ner', 'Content', 'Noun Phrase', 'Emotion', 'Mod&Aux', 'Compl&Obj']
data_plot = torch.cat((packed_feature.data[:, 0:2], predicted_feature[:, 0, :]), dim=1)
pnFix = predcited_nFix.detach().squeeze().tolist()

fig, axs = plt.subplots(figsize=(12, 6), nrows=2, sharex=False, gridspec_kw=dict(height_ratios=[0.5, 1]))
line1 = axs[0].plot(nFix, color='fuchsia', marker='^', linestyle='dashed', label='Target nFix')
line2 = axs[0].plot(pnFix, color='lightcoral', marker='o', linestyle='dashed', label='Prediction nFix')
lns = line1+line2
labs = [l.get_label() for l in lns]
axs[0].legend(loc='best')
axs[0].legend(lns, labs, loc='best')
axs[0].get_xaxis().set_visible(False)
axs[0].grid()
axs[0].set_ylabel('nFix(number of Fixations)')
sns.heatmap(data_plot.transpose(1, 0).detach(), cbar=False, ax=axs[1], cmap=sns.light_palette("#2ecc71", as_cmap=True), vmin=0, vmax=data_plot.transpose(1,0).detach().max(), xticklabels=words, yticklabels=feature_name, annot=True, square=False)
axs[1].xaxis.set_ticks_position('top')
axs[1].tick_params(left=False, top=False)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0, fontsize=9)
axs[1].set_yticklabels(axs[1].get_yticklabels(), fontsize=9)
#axs[1].set_title('Prediction Feature Weight', fontsize=12)
st = fig.suptitle("Example of predicted nFix and assigned with-in feature score", fontsize="large")
fig.tight_layout()
plt.savefig('lstm_prediction.pdf', bbox_inches='tight')
plt.show()
