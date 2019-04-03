import json
from nltk.tokenize import sent_tokenize
import numpy as np


def create_heatmap2():
    ifp_v = open('vocab.json', 'rb')

    vocab_map = json.load(ifp_v)

    src_list = []
    tgt_list = []
    attn_list = []
    src_list_raw = []
    num_pos = 0
    len_ls = []

    ofp_json = open('data.nosync/train/cnndm.json', 'w+')
    data = {'x' : [], 'y' : []}

    for i in xrange(2):
        ifp_model = open('stanford_attn' + str(i), 'rb')
        ifp_data = np.load(ifp_model)

        for sample in ifp_data:
            src_ls_sample = sample[0]
            tgt_ls_sample = sample[1]
            attn_ls_sample = sample[2]

            src_ls_raw = [vocab_map[str(x)] for x in src_ls_sample]

            src_list.append(src_ls_sample)
            tgt_list.append(tgt_ls_sample)
            attn_list.append(attn_ls_sample)
            src_list_raw.append(src_ls_raw)

        ifp_model.close()

    assert len(tgt_list) == len(src_list)
    for a_ls, x_ls, y_ls, x_ls_r in zip(attn_list, src_list, tgt_list, src_list_raw):

        assert len(x_ls) == len(x_ls_r)

        most_used_idxs_map = get_most_used(a_ls, y_ls)

        doc = ' '.join(x_ls_r).encode('utf-8')
        sentences = sent_tokenize(doc)
        token_idx = 0

        for sent in sentences:
            total_in_sent = 0
            token_begin = token_idx
            highlight_ls = []

            for i, _ in enumerate(sent.split()):
                if token_idx in most_used_idxs_map:
                    total_in_sent += 1
                    highlight_ls.append(1)
                else:
                    highlight_ls.append(0)

                token_idx += 1

            highlight_ls, longest_span = combine_chunks(highlight_ls)

            sentence_label = total_in_sent >= 3 and longest_span[1] - longest_span[0] + 1 > 5

            single_y = [int(sentence_label)]

            if sentence_label:
                single_y.append(longest_span[0])
                single_y.append(longest_span[1])

                num_pos += 1
            else:
                single_y.extend([-1, -1])

            data['x'].append(x_ls[token_begin:token_idx].tolist())
            data['y'].append(single_y)

            len_ls.append(len(data['x'][-1]))

    json.dump(data, ofp_json)
    ofp_json.close()

    print num_pos/float(len(data['y'])), len(data['y']), len(data['x'])
    print np.median(len_ls), np.mean(len_ls)


def get_most_used(a_ls, y_ls):
    most_used = dict()

    for i, sub_ls_a, y in zip(range(len(a_ls)), a_ls, y_ls):
        if y == '</s>' or y == '<blank>':
            continue
        most_used[np.argmax(sub_ls_a)] = i

    return most_used


def combine_chunks(highlight_ls):
    longest_span = (None, None)
    distance_from_hl = 0
    is_begin = True
    cur_len = 0

    for i in xrange(len(highlight_ls)):
        if highlight_ls[i] == 1:
            if is_begin and 0 < distance_from_hl < 5:
                for j in range(0, i):
                    highlight_ls[j] = 1
            elif 0 < distance_from_hl < 5:
                for j in range(i - distance_from_hl, i):
                    highlight_ls[j] = 1

            distance_from_hl = 0
            is_begin = False

        distance_from_hl += 1

    if distance_from_hl <= 5:
        for j in range(len(highlight_ls) - distance_from_hl, len(highlight_ls)):
            highlight_ls[j] = 1

    for i, item in enumerate(highlight_ls):
        if item == 1:
            cur_len += 1
        else:
            if longest_span[0] is None:
                longest_span = (i - cur_len, i - 1)
            if cur_len > longest_span[1] - longest_span[0] + 1:
                longest_span = (i - cur_len, i - 1)
            cur_len = 0

    if cur_len > 0 and longest_span[0] is None:
        longest_span = (len(highlight_ls) - 1 - cur_len, len(highlight_ls) - 1)
    elif cur_len > 0 and cur_len > longest_span[1] - longest_span[0] + 1:
        longest_span = (len(highlight_ls) - 1 - cur_len, len(highlight_ls) - 1)

    return highlight_ls, longest_span


create_heatmap2()