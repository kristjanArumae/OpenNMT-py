import json
from random import uniform

from pytorch_pretrained_bert import BertTokenizer


def tokenize_data(data_split='train', output_to_html=-1, small_subset=-1, balance=None, bert_model='bert-large-uncased'):
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    ifp = open('data.nosync/' + data_split + '/cnndm.json', 'rb')
    data = json.load(ifp)

    ifp.close()

    bal_str = ''

    if balance is not None and data_split == 'train': # do not balance test or valid
        bal_str = '_balance_' + str(balance).replace('.', '_') + '_'

    ofp_html = None

    if output_to_html > 0:
        ofp_html = open('data.nosync/' + data_split + '/' + bert_model + '_cnndm_bert' + bal_str + '.html', 'w+')

    sent_tokenized_ls = []
    updated_labels_ls = []
    updated_s_id_ls = []
    updated_b_id_ls = []
    sent_orig_ls = []
    sent_align_ls = []

    total = 0
    total_o = 0
    pos_lbl = 0

    for i, (sent, labels, sent_id, batch_id) in enumerate(zip(data['x_o'], data['y'], data['s_id'], data['batch_id'])):

        if i == small_subset:
            break

        total_o += 1

        word_ls = ['[CLS]'] + sent.split() + ['[SEP]']
        label_begin = labels[1] + 1 if labels[1] != -1 else -1
        label_end = labels[2] + 1 if labels[2] != -1 else -1
        location = 0

        sent_tokenized_as_idx = []
        sent_tokenized_as_tok = []
        sent_align = []
        moving_idx = 0

        for j, word in enumerate(word_ls):
            if moving_idx >= label_begin and location == 0:
                location += 1
            elif moving_idx > label_end and location == 1:
                location += 1

            tok = tokenizer.tokenize(word)
            x = tokenizer.convert_tokens_to_ids(tok)
            len_x = len(x)

            sent_align.extend([j] * len_x)

            if len_x > 1:
                move_amt = len_x - 1

                if location == 0:
                    label_begin += move_amt
                    label_end += move_amt
                elif location == 1:
                    label_end += move_amt

            sent_tokenized_as_idx.extend(x)
            sent_tokenized_as_tok.extend(tok)

            moving_idx += len_x

        if i < output_to_html:
            orig_str = []

            for j, w in enumerate(sent_tokenized_as_tok):
                if w == '<unk>':
                    w = 'UNK'
                if w == '<t>':
                    w = 'T_BEGIN'
                if w == '</t>':
                    w = 'T_END'
                if w == '<blank>':
                    w = 'PADDING'

                if label_begin <= j <= label_end and labels[0] > 0:
                    orig_str.append(
                        '<span class="tag" style="background-color: rgba(255, 0, 0, 0.7);">' + w + ' </span >')
                else:
                    orig_str.append('<span class="tag">' + w + ' </span >')

            ofp_html.write(' '.join(orig_str))
            ofp_html.write('</br>')

        if labels[0] < 1:
            if balance is not None:
                rand = uniform(0, 1)

                if rand > balance:
                    continue

            updated_labels_ls.append(labels)
        else:
            updated_labels_ls.append([1, label_begin, label_end])
            pos_lbl += 1

        sent_align_ls.append(sent_align)
        updated_s_id_ls.append(sent_id)
        updated_b_id_ls.append(batch_id)

        sent_tokenized_ls.append([sent_tokenized_as_idx, sent_tokenized_as_tok])
        sent_orig_ls.append(' '.join(word_ls))

        total += 1

    ofp = open('data.nosync/' + data_split + '/' + bert_model + '_cnndm_labeled_tokenized' + bal_str + '.json', 'w+')

    updated_data = dict()
    updated_data['x'] = sent_tokenized_ls
    updated_data['x_orig'] = sent_orig_ls
    updated_data['x_align'] = sent_align_ls
    updated_data['y'] = updated_labels_ls
    updated_data['s_id'] = updated_s_id_ls
    updated_data['b_id'] = updated_b_id_ls
    updated_data['rouge'] = data['rouge']

    print('original total : ', total_o)

    print('current :', pos_lbl, '/', total, '\n')

    json.dump(updated_data, ofp)

    ofp.close()

    if output_to_html > 0:
        ofp_html.close()


tokenize_data(data_split='train', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-large-uncased')
tokenize_data(data_split='train', output_to_html=10000, small_subset=-1, balance=0.33, bert_model='bert-large-uncased')
tokenize_data(data_split='valid', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-large-uncased')
tokenize_data(data_split='test', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-large-uncased')

tokenize_data(data_split='train', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-base-uncased')
tokenize_data(data_split='train', output_to_html=10000, small_subset=-1, balance=0.33, bert_model='bert-base-uncased')
tokenize_data(data_split='valid', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-base-uncased')
tokenize_data(data_split='test', output_to_html=10000, small_subset=-1, balance=None, bert_model='bert-base-uncased')
