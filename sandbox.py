import torch
from torch import nn
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_pretrained_bert import BertModel, BertAdam, BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import f1_score, accuracy_score


class CustomNetwork(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetwork, self).__init__(config)

        self.num_labels = num_labels
        config.type_vocab_size = config.max_position_embeddings
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

        self.dropout_qa = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_s = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        print('model loaded')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None,end_positions=None, weights=None, train=False):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        pooled_output = self.dropout_s(pooled_output)
        sequence_output = self.dropout_qa(sequence_output)

        logits = self.classifier(pooled_output)

        logits_qa = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits_qa.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if train:

            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)

            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)

            loss_sent = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            loss_qa = (start_loss + end_loss) / 10.0

            total_loss = loss_qa + loss_sent

            return total_loss, loss_sent, loss_qa
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)

            loss_sent = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            loss_qa = (start_loss + end_loss) / 10.0

            total_loss = loss_qa + loss_sent

            return torch.nn.functional.softmax(start_logits, dim=-1), torch.nn.functional.softmax(end_logits, dim=-1), torch.nn.functional.softmax(logits, dim=-1), total_loss


class CustomNetworkQA(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetworkQA, self).__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None,
                end_positions=None, weights=None):
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
        config.type_vocab_size = config.max_position_embeddings
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None,
                end_positions=None, weights=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def create_iterator(max_len=30, max_size=-1):
    ifp = open('data.nosync/train/cnndm_labeled_tokenized.json', 'rb')
    rouge_model_path = 'data.nosync/train/small_model/'

    if not os.path.exists(rouge_model_path):
        os.mkdir(rouge_model_path)

    data = json.load(ifp)

    ifp.close()

    x_ls, y_ls, s_idx_ls, b_id_ls, rouge_dict, x_for_rouge = data['x'], data['y'], data['s_id'], data['b_id'], data[
        'rouge'], data['x_orig']

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_start_positions = []
    all_end_positions = []
    all_sent_labels = []
    batch_id_list = []

    num_t = 0
    for (x, _), (label, start, end), s_id, b_id in zip(x_ls, y_ls, s_idx_ls, b_id_ls):

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

        segment_id = [s_id] * max_len

        all_segment_ids.append(segment_id[:max_len])
        batch_id_list.append(b_id)

        num_t += 1

        if num_t == max_size:
            break

    val_split = len(all_input_ids) // 20

    tensor_data_train = TensorDataset(torch.tensor(all_input_ids[val_split:], dtype=torch.long),
                                      torch.tensor(all_input_mask[val_split:], dtype=torch.long),
                                      torch.tensor(all_start_positions[val_split:], dtype=torch.long),
                                      torch.tensor(all_end_positions[val_split:], dtype=torch.long),
                                      torch.tensor(all_sent_labels[val_split:], dtype=torch.long),
                                      torch.tensor(all_segment_ids[val_split:], dtype=torch.long),
                                      torch.tensor(batch_id_list[val_split:], dtype=torch.long))

    tensor_data_valid = TensorDataset(torch.tensor(all_input_ids[:val_split], dtype=torch.long),
                                      torch.tensor(all_input_mask[:val_split], dtype=torch.long),
                                      torch.tensor(all_start_positions[:val_split], dtype=torch.long),
                                      torch.tensor(all_end_positions[:val_split], dtype=torch.long),
                                      torch.tensor(all_sent_labels[:val_split], dtype=torch.long),
                                      torch.tensor(all_segment_ids[:val_split], dtype=torch.long),
                                      torch.tensor(batch_id_list[:val_split], dtype=torch.long))

    used_b_id = dict()
    rouge_counter = 0

    for batch_id in batch_id_list[:val_split]:

        if batch_id not in used_b_id:

            y_text = rouge_dict[str(batch_id)]

            ofp_rouge = open(rouge_model_path + 'm_' + str(rouge_counter).zfill(6) + '.txt', 'w+')
            ofp_rouge.write(y_text)
            ofp_rouge.close()

            used_b_id[batch_id] = rouge_counter
            rouge_counter += 1

    return DataLoader(tensor_data_train, sampler=RandomSampler(tensor_data_train), batch_size=128), DataLoader(
        tensor_data_valid, batch_size=128), num_t, used_b_id, x_for_rouge[:val_split]


def get_valid_evaluation(eval_gt_start,
                         eval_gt_end,
                         eval_gt_sent,
                         eval_sys_start,
                         eval_sys_end,
                         eval_sys_sent):
    ooi = len(eval_sys_end[0])

    updated_eval_gt_start = []
    updated_eval_gt_end = []

    updated_eval_sys_start = []
    updated_eval_sys_end = []

    for g, s in zip(eval_gt_start, eval_sys_start):
        if g < ooi:
            updated_eval_gt_start.append(g)
            updated_eval_sys_start.append(s)

    for g, s in zip(eval_gt_end, eval_sys_end):
        if g < ooi:
            updated_eval_gt_end.append(g)
            updated_eval_sys_end.append(s)

    start_f1 = f1_score(updated_eval_gt_start, np.argmax(updated_eval_sys_start, axis=1), average='micro')
    end_f1 = f1_score(updated_eval_gt_end, np.argmax(updated_eval_sys_end, axis=1), average='micro')

    sent_f1 = f1_score(eval_gt_sent, np.argmax(eval_sys_sent, axis=1))

    start_acc = accuracy_score(updated_eval_gt_start, np.argmax(updated_eval_sys_start, axis=1))
    end_acc = accuracy_score(updated_eval_gt_end, np.argmax(updated_eval_sys_end, axis=1))

    acc_sent = accuracy_score(eval_gt_sent, np.argmax(eval_sys_sent, axis=1))

    return (start_acc + end_acc) / 2.0, (start_f1 + end_f1) / 2.0, acc_sent, sent_f1
    # return acc_sent, sent_f1


def create_valid_rouge(rouge_dict, x_for_rouge, eval_sys_sent, batch_ids):
    rouge_sys_sent_path = 'data.nosync/train/small_sys_sent/'

    ofp_rouge = None
    cur_batch = -1

    for x_o, sys_lbl, b_id in zip(x_for_rouge, eval_sys_sent, batch_ids):

        if cur_batch != b_id:
            cur_batch = b_id

            if ofp_rouge is not None:
                ofp_rouge.close()

            ofp_rouge = open(rouge_sys_sent_path + 's_' + str(rouge_dict[cur_batch]).zfill(6) + '.txt', 'w+')

            if sys_lbl[1] > sys_lbl[0]:
                ofp_rouge.write(x_o)
                ofp_rouge.write(' ')

        elif sys_lbl[1] > sys_lbl[0]:
            ofp_rouge.write(x_o)
            ofp_rouge.write(' ')

    ofp_rouge.close()


def train(model, loader_train, loader_valid, num_examples, num_train_epochs=70, rouge_dict=None, x_for_rouge=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # num_train_optimization_steps = int(num_examples / 128)

    ofp_model = 'data.nosync/small_model.pt'

    rouge_sys_sent_path = 'data.nosync/train/small_sys_sent/'
    rouge_sys_segs_path = 'data.nosync/train/small_sys_segs/'

    if not os.path.exists(rouge_sys_sent_path):
        os.mkdir(rouge_sys_sent_path)
    if not os.path.exists(rouge_sys_segs_path):
        os.mkdir(rouge_sys_segs_path)

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # param_optimizer = list(model.named_parameters())

    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    optimizer = BertAdam(model.parameters(), lr=1e-05)

    model.train()

    loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls = [], [], [], []
    qa_acc, qa_f1, sent_acc, sent_f1 = [], [], [], []

    best_valid = 100.0
    unchanged = 0
    unchanged_limit = 30

    # weights = torch.tensor([0.01, 1.0], dtype=torch.float32).to(device)
    weights = None

    for _ in trange(num_train_epochs, desc="Epoch"):
        for step, batch in enumerate(tqdm(loader_train, desc="Iteration")):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids, _ = batch

            loss, loss_s, loss_q = model(input_ids, seg_ids, input_mask, sent_labels, start_positions, end_position, weights, train=True)

            loss.backward()
            optimizer.step()

            if (step + 1) % 300 == 0:
                loss_ls.append(float(loss.cpu().data.numpy()))
                loss_ls_s.append(float(loss_s.cpu().data.numpy()))
                loss_ls_qa.append(float(loss_q.cpu().data.numpy()))

                with torch.no_grad():
                    eval_gt_start, eval_gt_end, eval_gt_sent = [], [], []
                    eval_sys_start, eval_sys_end, eval_sys_sent = [], [], []

                    valid_ls = []

                    batch_ids = []
                    for _, batch_valid in enumerate(tqdm(loader_valid, desc="Validation")):
                        batch_valid = tuple(t2.to(device) for t2 in batch_valid)

                        input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids, batch_id = batch_valid
                        start_l, end_l, sent_l, valid_l = model(input_ids, seg_ids, input_mask, sent_labels, start_positions, end_position, None)
                        # sent_l = model(input_ids, seg_ids, input_mask, None, None, None)

                        eval_gt_start.extend(start_positions.cpu().data.numpy())
                        eval_gt_end.extend(end_position.cpu().data.numpy())
                        eval_gt_sent.extend(sent_labels.cpu().data.numpy())

                        eval_sys_start.extend(start_l.cpu().data.numpy())
                        eval_sys_end.extend(end_l.cpu().data.numpy())
                        eval_sys_sent.extend(sent_l.cpu().data.numpy())

                        batch_ids.extend(batch_id.cpu().data.numpy().tolist())

                        valid_ls.append(valid_l.cpu().data.numpy())

                    qa_acc_val, qa_f1_val, sent_acc_val, sent_f1_val = get_valid_evaluation(eval_gt_start,
                                                                                            eval_gt_end,
                                                                                            eval_gt_sent,
                                                                                            eval_sys_start,
                                                                                            eval_sys_end,
                                                                                            eval_sys_sent)

                    avg_val_loss = np.mean(valid_ls)

                    qa_acc.append(qa_acc_val)
                    qa_f1.append(qa_f1_val)
                    sent_acc.append(sent_acc_val)
                    sent_f1.append(sent_f1_val)
                    loss_valid_ls.append(avg_val_loss)

                    if avg_val_loss < best_valid:
                        best_valid = avg_val_loss
                        unchanged = 0

                        create_valid_rouge(rouge_dict, x_for_rouge, eval_sys_sent, batch_ids)

                    elif unchanged > unchanged_limit:

                        plt.plot([i for i in range(len(loss_ls))], loss_ls, '-', label="loss", linewidth=1)
                        plt.plot([i for i in range(len(loss_ls))], loss_ls_s, '-', label="sent", linewidth=1)
                        plt.plot([i for i in range(len(loss_ls))], loss_ls_qa, '-', label="qa", linewidth=1)
                        plt.plot([i for i in range(len(loss_ls))], loss_valid_ls, '-', label="valid", linewidth=1)

                        plt.legend(loc='best')
                        plt.savefig('loss_model.png', dpi=400)

                        plt.clf()

                        # plt.plot([i for i in range(len(qa_acc))], qa_acc, '-', label="qa acc", linewidth=1)
                        plt.plot([i for i in range(len(qa_acc))], qa_f1, '-', label="qa f1", linewidth=1)
                        # plt.plot([i for i in range(len(sent_acc))], sent_acc, '-', label="sent acc", linewidth=1)
                        plt.plot([i for i in range(len(sent_f1))], sent_f1, '-', label="sent f1", linewidth=1)

                        plt.legend(loc='best')
                        plt.savefig('val_model.png', dpi=400)

                        return
                    else:
                        unchanged += 1

    plt.plot([i for i in range(len(loss_ls))], loss_ls, '-', label="loss", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_s, '-', label="sent", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_qa, '-', label="qa", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_valid_ls, '-', label="valid", linewidth=1)

    plt.legend(loc='best')
    plt.savefig('loss_model.png', dpi=400)

    plt.clf()

    # plt.plot([i for i in range(len(qa_acc))], qa_acc, '-', label="qa acc", linewidth=1)
    plt.plot([i for i in range(len(qa_acc))], qa_f1, '-', label="qa f1", linewidth=1)
    # plt.plot([i for i in range(len(sent_acc))], sent_acc, '-', label="sent acc", linewidth=1)
    plt.plot([i for i in range(len(sent_f1))], sent_f1, '-', label="sent f1", linewidth=1)

    plt.legend(loc='best')
    plt.savefig('metrics_model.png', dpi=400)


loader_train_, loader_valid_, _n, rouge_map, x_for_rouge = create_iterator(max_size=100000)
print('loaded data', _n)
train(model=CustomNetwork.from_pretrained('bert-base-uncased'),
      loader_train=loader_train_,
      loader_valid=loader_valid_,
      num_examples=_n,
      num_train_epochs=75,
      rouge_dict=rouge_map,
      x_for_rouge=x_for_rouge)

