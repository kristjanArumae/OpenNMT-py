import torch
from torch import nn
import json
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from pytorch_pretrained_bert import BertModel, BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import train_args as parse
from m_io import create_output_name, create_valid_rouge, get_valid_evaluation, create_metric_figure

from tqdm import tqdm, trange

import numpy as np
import os


class CustomNetwork(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(CustomNetwork, self).__init__(config)

        self.num_labels = num_labels
        config.type_vocab_size = config.max_position_embeddings
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)

        self.dropout_qa = nn.Dropout(0.1)
        self.dropout_s = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

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


def create_iterator(data_split='train', max_len=45, max_size=-1, batch_size=32, balance=None, bert_model='bert-large-uncased', ofp_fname=''):
    bal_str = ''

    if balance is not None and data_split == 'train':  # do not balance test or valid
        bal_str = '_balance_' + str(balance).replace('.', '_') + '_'

    ifp = open('data.nosync/' + data_split + '/' + bert_model + '_cnndm_labeled_tokenized' + bal_str + '.json', 'rb')

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

    tensor_data = TensorDataset(torch.tensor(all_input_ids, dtype=torch.long),
                                torch.tensor(all_input_mask, dtype=torch.long),
                                torch.tensor(all_start_positions, dtype=torch.long),
                                torch.tensor(all_end_positions, dtype=torch.long),
                                torch.tensor(all_sent_labels, dtype=torch.long),
                                torch.tensor(all_segment_ids, dtype=torch.long),
                                torch.tensor(batch_id_list, dtype=torch.long))

    if data_split == 'train':
        sampler = RandomSampler(tensor_data)
        used_b_id = None
    else:
        sampler = None

        used_b_id = dict()
        rouge_counter = 0

        rouge_model_path = 'data.nosync/' + ofp_fname + '_rouge/'

        if not os.path.exists(rouge_model_path):
            os.mkdir(rouge_model_path)

        for batch_id in batch_id_list:

            if batch_id not in used_b_id:
                y_text = rouge_dict[str(batch_id)]

                ofp_rouge = open(rouge_model_path + 'm_' + str(rouge_counter).zfill(6) + '.txt', 'w+')
                ofp_rouge.write(y_text)
                ofp_rouge.close()

                used_b_id[batch_id] = rouge_counter
                rouge_counter += 1

    data_loader = DataLoader(tensor_data, sampler=sampler, batch_size=batch_size)

    return data_loader, num_t, used_b_id, x_for_rouge, all_sent_align


def train(model, loader_train, loader_valid, num_train_epochs=70, rouge_dict=None, x_for_rouge=None, x_sent_align=None, optim='adam', learning_rate=3e-5, unchanged_limit=20, weights=None, ofp_fname='PLT'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rouge_sys_sent_path = 'data.nosync/train/small_sys_sent_bert_large/'
    rouge_sys_segs_path = 'data.nosync/train/small_sys_segs_bert_large/'

    if not os.path.exists(rouge_sys_sent_path):
        os.mkdir(rouge_sys_sent_path)
    if not os.path.exists(rouge_sys_segs_path):
        os.mkdir(rouge_sys_segs_path)

    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = BertAdam(model.parameters(), lr=learning_rate)

    model.train()

    loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls = [], [], [], []
    qa_acc, qa_f1, sent_acc, sent_f1 = [], [], [], []

    acc_loss, acc_loss_s, acc_loss_qa = [], [], []

    best_valid = 1e3
    unchanged = 0

    if weights is not None:
        weights = torch.tensor([weights, 1.0], dtype=torch.float32).to(device)

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

            if (step + 1) % 1000 == 0:
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
                                                                                                 rouge_sys_segs_path,
                                                                                                 ofp_fname)

                    elif unchanged > unchanged_limit:
                        create_metric_figure(ofp_fname, loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls, qa_f1, sent_f1, cur_used_ls_mean, total_used, total_s, mean_seg_len)
                        return
                    else:
                        unchanged += 1

    create_metric_figure(ofp_fname, loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls, qa_f1, sent_f1, cur_used_ls_mean, total_used, total_s, mean_seg_len)


args = parse.get_args()

batch_size = args.batch_size
sent_len = args.sent_len

if args.train:

    ofp_fname = create_output_name(args)

    data_loader_train, num_train, _, _, _ = create_iterator(data_split='train',
                                                            max_len=sent_len,
                                                            max_size=1000,
                                                            batch_size=batch_size,
                                                            balance=args.balance,
                                                            bert_model=args.bert_model,
                                                            ofp_fname=ofp_fname)
    data_loader_valid, num_val, used_b_id, x_for_rouge, all_sent_align = create_iterator(data_split='valid',
                                                                                         max_len=sent_len,
                                                                                         max_size=128,
                                                                                         batch_size=batch_size,
                                                                                         balance=None,
                                                                                         bert_model=args.bert_model,
                                                                                         ofp_fname=ofp_fname)

    model = CustomNetwork.from_pretrained(args.bert_model)

    train(model=model,
          loader_train=data_loader_train,
          loader_valid=data_loader_valid,
          num_train_epochs=args.epochs,
          rouge_dict=used_b_id,
          x_for_rouge=x_for_rouge,
          x_sent_align=all_sent_align,
          optim=args.optim,
          learning_rate=args.lr,
          unchanged_limit=args.unchanged_limit,
          weights=args.weights,
          ofp_fname=ofp_fname)
else:
    raise NotImplementedError



