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

        self.dropout_qa = nn.Dropout(0.25)
        self.dropout_s = nn.Dropout(0.25)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        print('model loaded')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, start_positions=None,end_positions=None, weights=None, train=False):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

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
            ignored_index = start_logits.size(1)

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


def create_iterator(max_len=45, max_size=-1):
    ifp = open('data.nosync/train/cnndm_labeled_tokenized.json', 'rb')
    rouge_model_path = 'data.nosync/train/small_model/'

    if not os.path.exists(rouge_model_path):
        os.mkdir(rouge_model_path)

    data = json.load(ifp)

    ifp.close()

    x_ls, y_ls, s_idx_ls, b_id_ls, rouge_dict, x_for_rouge, x_align = data['x'], data['y'], data['s_id'], data['b_id'], data[
        'rouge'], data['x_orig'], data['x_align']

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_start_positions = []
    all_end_positions = []
    all_sent_labels = []
    all_sent_align = []
    batch_id_list = []

    num_t = 0
    for (x, _), (label, start, end), s_id, b_id, x_a in zip(x_ls, y_ls, s_idx_ls, b_id_ls, x_align):

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
        all_sent_align.append(x_a)

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
        tensor_data_valid, batch_size=128), num_t, used_b_id, x_for_rouge[:val_split], all_sent_align[:val_split]


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


def create_valid_rouge(rouge_dict, x_for_rouge, eval_sys_sent, eval_sys_start, eval_sys_end, gt_sent, gt_start, gt_end,
                       batch_ids, align_ls, rouge_sys_sent_path, rouge_sys_segs_path):

    ofp_rouge_sent = None
    ofp_rouge_segm = None

    cur_batch = -1

    used_set = set()

    total_s = 0
    total_used = 0
    cur_used = 0
    cur_used_ls = []

    uesd_seg_len = []

    ofp_readable = open('data.nosync/readable.html', 'w+')

    for x_o, sys_lbl_s, sys_lbl_start, sys_lbl_end, model_lbl_s, model_lbl_start, model_lbl_end, b_id, x_a in zip(
            x_for_rouge, eval_sys_sent, eval_sys_start, eval_sys_end, gt_sent, gt_start, gt_end, batch_ids, align_ls):
        total_s += 1
        assert b_id not in used_set

        start_idx = min(np.argmax(sys_lbl_start), x_a[-1])
        end_idx = min(np.argmax(sys_lbl_end), x_a[-1])

        if end_idx < start_idx:
            end_idx = min(start_idx + np.argmax(sys_lbl_start[start_idx:]), x_a[-1])

        start_idx_aligned = x_a[start_idx]
        end_idx_aligned = x_a[end_idx]

        if model_lbl_s > 0:
            start_idx_model = x_a[model_lbl_start] if model_lbl_start < len(x_a) else x_a[-1]
            end_idx_model = x_a[model_lbl_end] if model_lbl_end < len(x_a) else x_a[-1]
        else:
            start_idx_model = end_idx_model = -1

        if cur_batch != b_id:

            used_set.add(cur_batch)
            cur_batch = b_id

            if ofp_rouge_sent is not None:
                ofp_rouge_sent.close()
                ofp_rouge_segm.close()
                ofp_readable.write('</p>')

            ofp_readable.write('<p>')

            ofp_rouge_sent = open(rouge_sys_sent_path + 's_' + str(rouge_dict[cur_batch]).zfill(6) + '.txt', 'w+')
            ofp_rouge_segm = open(rouge_sys_segs_path + 's_' + str(rouge_dict[cur_batch]).zfill(6) + '.txt', 'w+')

            cur_used_ls.append(cur_used)
            cur_used = 0

            if sys_lbl_s[1] > sys_lbl_s[0]:
                segment = x_o.split()[start_idx_aligned:end_idx_aligned + 1]

                ofp_rouge_sent.write(x_o)
                ofp_rouge_segm.write(' '.join(segment))

                ofp_rouge_sent.write(' ')
                ofp_rouge_segm.write(' ')

                for i, token in enumerate(x_o.split()):
                    if model_lbl_s > 0:
                        if i < start_idx_aligned: # not started
                            if start_idx_model <= i <= end_idx_model:
                                ofp_readable.write('<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                            else:
                                ofp_readable.write(token + ' ')

                        elif start_idx_aligned <= i <= end_idx_aligned: # inside segment
                            if start_idx_model <= i <= end_idx_model:
                                ofp_readable.write(
                                    '<span style="background-color: rgba(0, 0, 255, 0.65);">' + token + ' </span>')
                            else:
                                ofp_readable.write(
                                    '<span style="background-color: rgba(255, 0, 0, 0.65);">' + token + ' </span>')
                        else: # after
                            if start_idx_model <= i <= end_idx_model:
                                ofp_readable.write('<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                            else:
                                ofp_readable.write(token + ' ')
                    else:
                        if i < start_idx_aligned: # not started
                            ofp_readable.write(token + ' ')
                        elif start_idx_aligned <= i <= end_idx_aligned: # inside segment
                            ofp_readable.write(
                                '<span style="background-color: rgba(255, 0, 0, 0.65);">' + token + ' </span>')
                        else: # after
                            ofp_readable.write(token + ' ')

                total_used += 1
                cur_used += 1

                uesd_seg_len.append(end_idx_aligned - start_idx_aligned)

                ofp_readable.write('</br>')
            else:
                if model_lbl_s > 0:
                    for i, token in enumerate(x_o.split()):
                        if start_idx_model <= i <= end_idx_model:
                            ofp_readable.write(
                                '<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                        else:
                            ofp_readable.write(token + ' ')

                    ofp_readable.write('</br>')

                else:
                    ofp_readable.write(x_o + '</br>')

        elif sys_lbl_s[1] > sys_lbl_s[0]:
            segment = x_o.split()[start_idx_aligned:end_idx_aligned + 1]

            ofp_rouge_sent.write(x_o)
            ofp_rouge_segm.write(' '.join(segment))

            ofp_rouge_sent.write(' ')
            ofp_rouge_segm.write(' ')

            for i, token in enumerate(x_o.split()):
                if model_lbl_s > 0:
                    if i < start_idx_aligned:  # not started
                        if start_idx_model <= i <= end_idx_model:
                            ofp_readable.write(
                                '<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                        else:
                            ofp_readable.write(token + ' ')

                    elif start_idx_aligned <= i <= end_idx_aligned:  # inside segment
                        if start_idx_model <= i <= end_idx_model:
                            ofp_readable.write(
                                '<span style="background-color: rgba(0, 0, 255, 0.65);">' + token + ' </span>')
                        else:
                            ofp_readable.write(
                                '<span style="background-color: rgba(255, 0, 0, 0.65);">' + token + ' </span>')
                    else:  # after
                        if start_idx_model <= i <= end_idx_model:
                            ofp_readable.write(
                                '<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                        else:
                            ofp_readable.write(token + ' ')
                else:
                    if i < start_idx_aligned:  # not started
                        ofp_readable.write(token + ' ')
                    elif start_idx_aligned <= i <= end_idx_aligned:  # inside segment
                        ofp_readable.write(
                            '<span style="background-color: rgba(255, 0, 0, 0.65);">' + token + ' </span>')
                    else:  # after
                        ofp_readable.write(token + ' ')

            ofp_readable.write('</br>')

            total_used += 1
            cur_used += 1

            uesd_seg_len.append(end_idx_aligned - start_idx_aligned)
        else:
            if model_lbl_s > 0:
                for i, token in enumerate(x_o.split()):
                    if start_idx_model <= i <= end_idx_model:
                        ofp_readable.write(
                            '<span style="background-color: rgba(0, 255, 0, 0.65);">' + token + ' </span>')
                    else:
                        ofp_readable.write(token + ' ')

                ofp_readable.write('</br>')

            else:
                ofp_readable.write(x_o + '</br>')

    ofp_rouge_sent.close()
    ofp_rouge_segm.close()
    ofp_readable.close()

    return np.mean(cur_used_ls), total_used, total_s, np.mean(uesd_seg_len)


def train(model, loader_train, loader_valid, num_examples, num_train_epochs=70, rouge_dict=None, x_for_rouge=None, x_sent_align=None, optim='adam'):
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

    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.01)
    else:
        optimizer = BertAdam(model.parameters(), lr=1e-06, weight_decay=0.01)

    model.train()

    loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls = [], [], [], []
    qa_acc, qa_f1, sent_acc, sent_f1 = [], [], [], []

    acc_loss, acc_loss_s, acc_loss_qa = [], [], []

    best_valid = 100.0
    unchanged = 0
    unchanged_limit = 20

    weights = torch.tensor([0.05, 1.0], dtype=torch.float32).to(device)
    # weights = None
    cur_used_ls_mean, total_used, total_s, mean_seg_len = None, None, None, None

    for _ in trange(num_train_epochs, desc="Epoch"):
        for step, batch in enumerate(tqdm(loader_train, desc="Iteration")):
            optimizer.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, start_positions, end_position, sent_labels, seg_ids, _ = batch

            loss, loss_s, loss_q = model(input_ids, seg_ids, input_mask, sent_labels, start_positions, end_position, weights, train=True)

            loss.backward()
            optimizer.step()

            acc_loss.append(loss.cpu().data.numpy())
            acc_loss_s.append(loss_s.cpu().data.numpy())
            acc_loss_qa.append(loss_q.cpu().data.numpy())

            if (step + 1) % 200 == 0:
                loss_ls.append(np.mean(acc_loss))
                loss_ls_s.append(np.mean(acc_loss_s))
                loss_ls_qa.append(np.mean(acc_loss_qa))

                acc_loss, acc_loss_s, acc_loss_qa = [], [], []

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

                        cur_used_ls_mean, total_used, total_s, mean_seg_len = create_valid_rouge(rouge_dict,
                                                                                                 x_for_rouge,
                                                                                                 eval_sys_sent,
                                                                                                 eval_sys_start,
                                                                                                 eval_sys_end,
                                                                                                 eval_gt_sent,
                                                                                                 eval_gt_start,
                                                                                                 eval_gt_end,
                                                                                                 batch_ids,
                                                                                                 x_sent_align,
                                                                                                 rouge_sys_sent_path,
                                                                                                 rouge_sys_segs_path)

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

                        print('\n\n\nSent used:', total_used, '/', total_s, total_used / float(total_s))
                        print('Avg len (sent)', cur_used_ls_mean)
                        print('avg seg len', mean_seg_len)

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


loader_train_, loader_valid_, _n, rouge_map, x_for_rouge, x_sent_align = create_iterator(max_size=500000)
print('loaded data', _n)
train(model=CustomNetwork.from_pretrained('bert-base-uncased'),
      loader_train=loader_train_,
      loader_valid=loader_valid_,
      num_examples=_n,
      num_train_epochs=200,
      rouge_dict=rouge_map,
      x_for_rouge=x_for_rouge,
      x_sent_align=x_sent_align,
      optim='adam')

