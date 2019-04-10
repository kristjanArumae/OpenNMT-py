import torch
from torch import nn
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_pretrained_bert import BertModel, BertAdam, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from tqdm import tqdm, trange

import matplotlib.pyplot as plt


class CustomNetwork(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetwork, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        logits_qa = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits_qa.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:

            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct_qa = nn.CrossEntropyLoss(ignore_index=-1)
            loss_fct_sent = nn.CrossEntropyLoss()

            loss_sent = loss_fct_sent(logits.view(-1, self.num_labels), labels.view(-1))

            start_loss = loss_fct_qa(start_logits, start_positions)
            end_loss = loss_fct_qa(end_logits, end_positions)

            loss_qa = (start_loss + end_loss) / 2

            total_loss = loss_qa + loss_sent

            return total_loss
        else:
            return start_logits, end_logits, logits


def create_iterator(max_len=30):
    ifp = open('data.nosync/train/cnndm_labeled_tokenized.json', 'rb')
    data = json.load(ifp)

    ifp.close()

    x_ls, y_ls = data['x'], data['y']

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_start_positions = []
    all_end_positions = []
    all_sent_labels = []

    for (x, _), (label, start, end) in zip(x_ls, y_ls):

        if start >= max_len:
            label = 0
            start = -1
            end = -1

        if end >= max_len:
            end = max_len - 1

        all_sent_labels.append(label)

        all_start_positions.append(start)
        all_end_positions.append(end)

        mask = [1] * len(x)
        padding_mask = [0] * (max_len - len(x))

        mask.extend(padding_mask)
        x.extend(padding_mask)

        all_input_ids.append(x[:max_len])
        all_input_mask.append(mask[:max_len])

        segment_id = [0] * max_len
        if label > 0:
            for i in range(start, end):
                segment_id[i] = 1

        all_segment_ids.append(segment_id[:max_len])

    tensor_data = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long), torch.tensor(all_input_mask, dtype=torch.long), torch.tensor(
        all_segment_ids, dtype=torch.long), torch.tensor(all_start_positions, dtype=torch.long), torch.tensor(
        all_end_positions, dtype=torch.long), torch.tensor(all_sent_labels, dtype=torch.long))

    return DataLoader(tensor_data, sampler=RandomSampler(tensor_data), batch_size=32), len(y_ls)


config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
       num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

num_train_epochs = 100
loader, num_examples = create_iterator()
print('loaded data')

model = CustomNetwork.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_train_optimization_steps = int(num_examples / 32)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

param_optimizer = list(model.named_parameters())

optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-05, warmup=0.1, t_total=num_train_optimization_steps)

model.train()
loss_ls = []
for _ in trange(num_train_epochs, desc="Epoch"):
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_positions, end_position, sent_labels = batch
        loss = model(input_ids, segment_ids, input_mask, sent_labels, start_positions, end_position)

        loss.backward()
        loss_ls.append(float(loss.data.numpy()))
        if (step + 1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

plt.plot([i for i in range(len(loss_ls))], loss_ls, '.-', ls='dashed', linewidth=2.5)
plt.savefig('ranges2.png', dpi = 400)