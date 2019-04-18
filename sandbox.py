import torch
from torch import nn
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_pretrained_bert import BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from tqdm import tqdm, trange

import matplotlib.pyplot as plt


class CustomNetwork(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetwork, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.dropout_qa = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_s = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None, weights=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.dropout_s(pooled_output)
        sequence_output = self.dropout_qa(sequence_output)

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

            loss_fct_qa = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss_fct_sent = nn.CrossEntropyLoss(weight=weights)

            loss_sent = loss_fct_sent(logits.view(-1, self.num_labels), labels.view(-1))

            start_loss = loss_fct_qa(start_logits, start_positions)
            end_loss = loss_fct_qa(end_logits, end_positions)

            loss_qa = (start_loss + end_loss) / 10

            total_loss = loss_qa + loss_sent

            return total_loss, loss_sent, loss_qa
        else:
            return start_logits, end_logits, logits


class CustomNetworkQA(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetworkQA, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None, weights=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class CustomNetworkSent(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetworkSent, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None, weights=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


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

    val_split = len(y_ls)//10

    for (x, _), (label, start, end) in zip(x_ls, y_ls):

        if start >= max_len or label == 0:
            label = 0
            start = max_len
            end = max_len

        if end > max_len:
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

    tensor_data_train = TensorDataset(torch.tensor(all_input_ids[val_split:], dtype=torch.long),
                                      torch.tensor(all_input_mask[val_split:], dtype=torch.long),
                                      torch.tensor(all_start_positions[val_split:], dtype=torch.long),
                                      torch.tensor(all_end_positions[val_split:], dtype=torch.long),
                                      torch.tensor(all_sent_labels[val_split:], dtype=torch.long))

    tensor_data_valid = TensorDataset(torch.tensor(all_input_ids[:val_split], dtype=torch.long),
                                      torch.tensor(all_input_mask[:val_split], dtype=torch.long),
                                      torch.tensor(all_start_positions[:val_split], dtype=torch.long),
                                      torch.tensor(all_end_positions[:val_split], dtype=torch.long),
                                      torch.tensor(all_sent_labels[:val_split], dtype=torch.long))

    return DataLoader(tensor_data_train, sampler=RandomSampler(tensor_data_train), batch_size=32),  DataLoader(tensor_data_valid, batch_size=32), len(y_ls)


def train(model, loader_train, loader_valid, num_examples, num_train_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_train_optimization_steps = int(num_examples / 32)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=1e-06, warmup=0.1, t_total=num_train_optimization_steps)

    model.train()
    loss_ls, loss_ls_s, loss_ls_qa = [], [], []
    best_loss = 100.0

    weights = torch.tensor([0.01, 1.0], dtype=torch.float32).to(device)

    for _ in trange(num_train_epochs, desc="Epoch"):
        for step, batch in enumerate(tqdm(loader_train, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, start_positions, end_position, sent_labels = batch

            loss, loss_s, loss_q = model(input_ids, None, input_mask, sent_labels, start_positions, end_position, weights)

            loss.backward()

            loss_ls.append(float(loss.cpu().data.numpy()))
            loss_ls_s.append(float(loss_s.cpu().data.numpy()))
            loss_ls_qa.append(float(loss_q.cpu().data.numpy()))

            if (step + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            continue
            loss_valid = None

            for _, batch_valid in enumerate(tqdm(loader_valid, desc="Validation")):
                batch_valid = tuple(t2.to(device) for t2 in batch_valid)

                input_ids, input_mask, start_positions, end_position, sent_labels = batch_valid
                loss_, _, _ = model(input_ids, None, input_mask, sent_labels, start_positions, end_position, weights)

                if loss_valid is None:
                    loss_valid = loss_
                else:
                    loss_valid += loss_

            loss_valid = float(loss_valid.cpu().data.numpy())

            if loss_valid < best_loss:
                best_loss = loss_valid
            else:
                plt.plot([i for i in range(len(loss_ls))], loss_ls, '-',  label="loss", linewidth=1)
                plt.plot([i for i in range(len(loss_ls))], loss_ls_s, '-', label="sent", linewidth=1)
                plt.plot([i for i in range(len(loss_ls))], loss_ls_qa, '-', label="qa", linewidth=1)

                plt.legend(loc='best')
                plt.savefig('ranges2.png', dpi=400)

                break

    plt.plot([i for i in range(len(loss_ls))], loss_ls, '-', label="loss", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_s, '-', label="sent", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_qa, '-', label="qa", linewidth=1)

    plt.legend(loc='best')
    plt.savefig('ranges2.png', dpi=400)

loader_train_, loader_valid_, n = create_iterator()
print('loaded data')

train(CustomNetwork.from_pretrained('bert-base-uncased'), loader_train_, loader_valid_, n)




