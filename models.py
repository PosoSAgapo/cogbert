import torch
from transformers.models.bert.modeling_bert import *
import pdb

class BayesianAttention(torch.nn.Module):#this module incorporates attention
    def __init__(self, config):
        super(BayesianAttention, self).__init__()
        self.config =config
        self.activation = torch.nn.Tanh()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.reg = torch.nn.Linear(config.hidden_size, config.num_attention_heads)
        self.scale_func = config.scale_func
        self.low_trands_cnn = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.feature_size = 8
        self.shape_size = torch.nn.Linear(1,  self.feature_size, bias=False)
        self.deduction_size = torch.nn.Linear(config.hidden_size,  self.feature_size)
        self.actv_func = torch.nn.GELU()
        self.low_trands_cnn = torch.nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.up_trands_cnn = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding='same', bias=False)
        #self.up_trands_cnn = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding='same')

    def forward(self, hidden_states, attention_matrix_word_length, attention_matrix_pos_tag, attention_matrix_ner,
                attention_matrix_emotion, attention_matrix_npchunk, attention_matrix_word_position,
                attention_matrix_syntax_dac, attention_matrix_syntax_pcob, readability, layer_idx):
        # Attention = attention_matrix_word_length + attention_matrix_pos_tag + attention_matrix_ner + \
        #             attention_matrix_emotion + attention_matrix_npchunk + attention_matrix_word_position
        batch_size = attention_matrix_npchunk.shape[0]
        readability_attention_score = self.activation(self.shape_size(torch.unsqueeze(readability, dim=1)))+1.5
        context_attention_score = self.activation(self.deduction_size(hidden_states)[:,0,:]) + 1.5
        assigned_attention = 1/(readability_attention_score*context_attention_score*torch.sqrt(torch.tensor(2*math.pi)))
        scaled_attention = assigned_attention / torch.unsqueeze(torch.max(assigned_attention, dim=1)[0],dim=1)
        attention_matrix_word_length = attention_matrix_word_length * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:, 0], dim=1), dim=2),dim=3)
        attention_matrix_pos_tag = attention_matrix_pos_tag * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:, 1], dim=1), dim=2),dim=3)
        attention_matrix_ner = attention_matrix_ner * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:,  2], dim=1), dim=2),dim=3)
        attention_matrix_word_position = attention_matrix_word_position * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:, 3], dim=1), dim=2),dim=3)
        attention_matrix_emotion = attention_matrix_emotion * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:, 4], dim=1), dim=2),dim=3)
        attention_matrix_npchunk = attention_matrix_npchunk * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(scaled_attention[:, 5], dim=1), dim=2),dim=3)
        if layer_idx <= 4:
            stack_dianogal_attention = torch.stack((attention_matrix_word_length, attention_matrix_pos_tag,
                                        attention_matrix_ner, attention_matrix_word_position), dim=1)
            Attention = torch.squeeze(torch.diag_embed(
                self.low_trands_cnn(torch.diagonal(stack_dianogal_attention, dim1=-1, dim2=2)[:, :, :, 0]))). \
                repeat(self.config.num_attention_heads, 1, 1, 1).transpose(0, 1)
        else:
            high_trands_stack = torch.stack((attention_matrix_npchunk, attention_matrix_emotion,
                                            attention_matrix_syntax_dac, attention_matrix_syntax_pcob), dim=1)
            Attention = self.up_trands_cnn(high_trands_stack[:, :, 0, :, :])
            Attention = Attention.repeat(1, self.config.num_attention_heads, 1, 1)
            #npchunk_attention = self.actv_func(self.up_trands_cnn(attention_matrix_npchunk[:,0:1,:,:]))
            #emotion_attention = self.actv_func(self.up_trands_cnn(attention_matrix_emotion[:,0:1,:,:]))
            #Attention = self.actv_func(torch.unsqueeze(torch.bmm(torch.squeeze(npchunk_attention),torch.squeeze(emotion_attention)),dim=1).repeat(1,self.config.num_attention_heads,1,1))
            #Attention = torch.unsqueeze(torch.bmm(torch.squeeze(npchunk_attention),torch.squeeze(emotion_attention)),dim=1).repeat(1,self.config.num_attention_heads,1,1)

            # syntax_dac_attention = self.actv_func(self.up_trands_cnn_syntax(attention_matrix_syntax_dac[:, 0:1, :, :]))
            # syntax_pcob_attention = self.actv_func(self.up_trands_cnn_syntax(attention_matrix_syntax_pcob[:, 0:1, :, :]))
            # pdb.set_trace()
            # Attention_2 = self.actv_func(torch.unsqueeze(torch.bmm(torch.squeeze(syntax_dac_attention),torch.squeeze(syntax_pcob_attention)),dim=1).repeat(1,self.config.num_attention_heads,1,1))
            # Attention_2 = torch.unsqueeze(torch.bmm(torch.squeeze(syntax_dac_attention),torch.squeeze(syntax_pcob_attention)),dim=1).repeat(1,self.config.num_attention_heads,1,1)
            # Attention = Attention_1 + Attention_2
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        reg_output = self.reg(pooled_output)
        reg_output = reg_output.transpose(-1, -2)
        reg_output = reg_output[:, :, :, None]
        if self.scale_func == 'tanh':
            # scale_result = 0.5 + 0.5 * torch.tanh(reg_output - Attention)
            scale_result = 0.5 + 0.5 * torch.tanh(Attention)
        elif self.scale_func == 'sigmoid':
            scale_result = torch.sigmoid(reg_output - Attention)
        return scale_result, scaled_attention

class AttentionSelector(nn.Module):
    def __init__(self, config):
        super(AttentionSelector, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.num_attention_heads)

    def forward(self, hidden_states):
        #first_token_tensor = hidden_states[:, 0]
        # [batch_size, seq_length, hiddden_size] -> [batch_size, num_heads, seq_length]
        selector_tensor = self.dense(hidden_states)
        return torch.sigmoid(selector_tensor).transpose(-1, -2)

class BayesianBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            scaled_attention_mask=None,
            attention_mask_dict=None,
            selector_outputs=None,
            past_key_value=None,
            output_attentions=False,
            layer_idx=None
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        if (scaled_attention_mask is not None) and (selector_outputs is not None):
            attention_scores_l = attention_scores * scaled_attention_mask
            #attention_scores_l = attention_scores_l  #attention_mask
            attention_scores_l = attention_scores_l + attention_mask #attention_mask
            attention_probs_l = nn.Softmax(dim=-1)(attention_scores_l)
            attention_probs_l = self.dropout(attention_probs_l)
            local_context_layer = torch.matmul(attention_probs_l, value_layer)
            context_layer = selector_outputs * local_context_layer + (1 - selector_outputs) * context_layer
            attention_probs = selector_outputs * attention_probs_l + (1 - selector_outputs) * attention_probs
        else:
            attention_probs = attention_probs
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BayesianBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BayesianBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, scaled_attention_mask=None, attention_mask_dict=None,
                selector_outputs=None, past_key_value=None, output_attentions=False, layer_idx=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            scaled_attention_mask,
            attention_mask_dict,
            selector_outputs,
            past_key_value,
            output_attentions,
            layer_idx
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BayesianBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BayesianBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    def forward(self,hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, scaled_attention_mask=None, attention_mask_dict=None,
                selector_outputs=None, past_key_value=None, output_attentions=False, layer_idx=None):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            scaled_attention_mask=scaled_attention_mask,
            attention_mask_dict = attention_mask_dict,
            selector_outputs=selector_outputs,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            layer_idx = layer_idx
        )
        attention_output = self_attention_outputs[0]
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

#this module is to encode attention
class BayesianBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BayesianBertEncoder, self).__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BayesianBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.scale_layer = nn.ModuleList([BayesianAttention(config) for _ in range(config.num_hidden_layers)])
        self.selector = nn.ModuleList([AttentionSelector(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        attention_matrix_dict = None,
        attention_matrix_mask_dict = None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        # set attention matrix
        attention_matrix_word_length = attention_matrix_dict['word_length'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_pos_tag = attention_matrix_dict['pos_tag'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_ner = attention_matrix_dict['ner'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_emotion = attention_matrix_dict['emotion'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_npchunk = attention_matrix_dict['npchunk'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_word_position = attention_matrix_dict['word_position'][:, None, :, :].\
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_syntax_dac = attention_matrix_dict['word_position'][:, None, :, :]. \
            expand(-1, self.config.num_attention_heads, -1, -1)
        attention_matrix_syntax_pcob = attention_matrix_dict['word_position'][:, None, :, :]. \
            expand(-1, self.config.num_attention_heads, -1, -1)
        #
        # attention_matrix_syntax_dac = attention_matrix_dict['syntax_dac'][:, None, :, :]. \
        #     expand(-1, self.config.num_attention_heads, -1, -1)
        # attention_matrix_syntax_pcob = attention_matrix_dict['syntax_pcob'][:, None, :, :]. \
        #     expand(-1, self.config.num_attention_heads, -1, -1)
        readability = attention_matrix_dict['readability']
        for i, (layer_module, scale_module, selector_module) in enumerate(zip(self.layer, self.scale_layer, self.selector)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # the code to incorporate the attention matrix
                scaled_attention, attention_weight = scale_module(hidden_states, attention_matrix_word_length,
                                                     attention_matrix_pos_tag, attention_matrix_ner,
                                                     attention_matrix_emotion, attention_matrix_npchunk,
                                                     attention_matrix_word_position, attention_matrix_syntax_dac,
                                                     attention_matrix_syntax_pcob, readability, i)
                selector_outputs = selector_module(hidden_states)
                extended_selector_outputs = selector_outputs[:, :, :, None]
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    scaled_attention,
                    attention_matrix_mask_dict,
                    extended_selector_outputs,
                    past_key_value,
                    output_attentions,
                    i
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), attention_weight

#this is to place Encoder to BertModel
class BayesianBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BayesianBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            attention_matrix_dict=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if attention_matrix_dict is not None:
            attention_matrix_mask_dict = {}
            # extended_diagonal_attention_mask = attention_matrix_dict['word_length'].clone()
            # extended_diagonal_attention_mask[extended_diagonal_attention_mask!=0] = 1
            # extended_diagonal_attention_mask = extended_diagonal_attention_mask[:, None, :, :]
            # extended_diagonal_attention_mask = extended_diagonal_attention_mask.to(dtype=next(self.parameters()).dtype)
            # extended_diagonal_attention_mask = (1.0 -extended_diagonal_attention_mask) * -10000.0
            # attention_matrix_mask_dict['diagonal'] = extended_diagonal_attention_mask
            # extended_npchunk_attention_mask = attention_matrix_dict['npchunk'].clone()
            # extended_npchunk_emotion_mask = attention_matrix_dict['emotion'].clone()
            # extended_npchunk_attention_mask[(extended_npchunk_attention_mask+extended_npchunk_emotion_mask)!=0] = 1
            # extended_npchunk_attention_mask = extended_npchunk_attention_mask[:, None, :, :]
            # extended_npchunk_attention_mask = extended_npchunk_attention_mask.to(dtype=next(self.parameters()).dtype)
            # extended_npchunk_attention_mask = (1.0 -extended_npchunk_attention_mask) * -10000.0
            # attention_matrix_mask_dict['ensemble'] = extended_npchunk_attention_mask
        else:
            attention_matrix_mask_dict = {}
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs, attention_weight = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            attention_matrix_dict=attention_matrix_dict,
            attention_matrix_mask_dict=attention_matrix_mask_dict,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), attention_weight

#modified Bertmodel to be suitable for BayesianBert
class BayesianBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BayesianBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.set_reg_weights()###initializing model weights

    def set_reg_weights(self):
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            attention_matrix_dict = None,
            uid_weight = 0,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs, attention_weight = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            attention_matrix_dict=attention_matrix_dict,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        final_attention_loss = None
        final_layer_attention = outputs.attentions[-1] if output_attentions else None
        try:
            final_layer_entropy = torch.distributions.Categorical(final_layer_attention).entropy() if output_attentions else None
            final_attention_loss = torch.mean(torch.mean(torch.std(final_layer_entropy, dim=2), dim=1))*uid_weight if output_attentions else None
        except ValueError:#some times the attention value will be zero, this is due to the reason of scaled attention mask
            final_attention_loss = torch.tensor(0)
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze()) + final_attention_loss
                else:
                    loss = loss_fct(logits, labels) + final_attention_loss
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + final_attention_loss
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels) + final_attention_loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), attention_weight

class BayesianBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BayesianBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        self.set_reg_weights()  ###initializing model weights

    def forward( self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, attention_matrix_dict=None, start_positions=None, end_positions=None,
                 output_attentions=None, output_hidden_states=None, return_dict=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            attention_matrix_dict=attention_matrix_dict,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        final_layer_attention = outputs.attentions[-1] if output_attentions else None
        final_layer_entropy = torch.distributions.Categorical(
            final_layer_attention).entropy() if output_attentions else None
        final_attention_loss = torch.mean(
            torch.mean(torch.std(final_layer_entropy, dim=2), dim=1)) if output_attentions else None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2 + final_attention_loss

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BayesianBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BayesianBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, attention_matrix_dict=None, uid_weight=0, labels=None, output_attentions=None,
            output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs, attention_weight = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            attention_matrix_dict=attention_matrix_dict,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        final_attention_loss = None
        final_layer_attention = outputs.attentions[-1] if output_attentions else None
        try:
            final_layer_entropy = torch.distributions.Categorical(final_layer_attention).entropy() if output_attentions else None
            final_attention_loss = torch.mean(torch.mean(torch.std(final_layer_entropy, dim=2), dim=1))*uid_weight if output_attentions else None
        except ValueError:#some times the attention value will be zero, this is due to the reason of scaled attention mask
            final_attention_loss = torch.tensor(0).cuda()
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels) + final_attention_loss
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + final_attention_loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), attention_weight

class BayesianBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
            super().__init__(config)

            self.bert = BayesianBertModel(config)
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, 1)

            self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,  head_mask=None,
            inputs_embeds=None, attention_matrix_dict=None, uid_weight=0,  labels=None, output_attentions=None,
            output_hidden_states=None, return_dict=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            attention_matrix_dict=attention_matrix_dict,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        final_attention_loss = None
        final_layer_attention = outputs.attentions[-1] if output_attentions else None
        reshaped_logits = logits.view(-1, num_choices)
        try:
            final_layer_entropy = torch.distributions.Categorical(final_layer_attention).entropy() if output_attentions else None
            final_attention_loss = torch.mean(torch.mean(torch.std(final_layer_entropy, dim=2), dim=1))*uid_weight if output_attentions else None
        except ValueError:#some times the attention value will be zero, this is due to the reason of scaled attention mask
            final_attention_loss = torch.tensor(0).cuda()
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels) + final_attention_loss

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class EyetrackRegressionBiLSTM(torch.nn.Module):
    def __init__(self, input_size=300, hidden_size=256, linear_output_size=1):
        super(EyetrackRegressionBiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_output_size = linear_output_size
        self.BiLSTM = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, proj_size=6, batch_first=True, num_layers=2, dropout=0.15)
        self.linear = torch.nn.Linear(in_features=8, out_features=1)
        self.relu = torch.nn.ReLU()
    def forward(self, embedding, mask, feature, data_idx):
        packed_output, (hidden_states, cell_states) = self.BiLSTM(embedding)
        unpacked_output, unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        unpacked_mask, unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(mask)
        unpacked_feature, unpacked_sentence_length = torch.nn.utils.rnn.pad_packed_sequence(feature)
        return_output = unpacked_output*unpacked_mask[:, :, 2:]
        predicted_nFix = self.relu(self.linear(torch.cat((unpacked_feature[:, :, 0:2], return_output), dim=2)))
        return predicted_nFix, return_output, unpacked_sentence_length
        #return unpacked_output, unpacked_sentence_length
