import json
from nltk.tokenize import sent_tokenize
import numpy as np
import io


def create_labels():
    ifp_v = open('vocab.json', 'rb')

    vocab_map = json.load(ifp_v)

    src_list = []
    tgt_list = []
    attn_list = []
    src_list_raw = []
    len_ls = []
    batch_idx = []

    num_pos = 0

    ofp_json = open('data.nosync/train/cnndm.json', 'w+')
    data = {'x': [], 'x_o': [], 'y': []}

    for i in xrange(2):
        ifp_model = open('stanford_attn' + str(i), 'rb')
        ifp_data = np.load(ifp_model)

        for sample in ifp_data:
            src_ls_sample = sample[0]
            tgt_ls_sample = sample[1]
            attn_ls_sample = sample[2]
            batch_idx_sample = sample[3]

            src_ls_raw = [vocab_map[str(x)] for x in src_ls_sample]

            src_list.append(src_ls_sample)
            tgt_list.append(tgt_ls_sample)
            attn_list.append(attn_ls_sample)
            src_list_raw.append(src_ls_raw)
            batch_idx.append(batch_idx_sample)

        ifp_model.close()

    # ifp_orig = open('data.nosync/train.txt.src', 'rb')
    ifp_orig = io.open("data.nosync/train.txt.src", mode="r", encoding="utf-8")
    x_orig, y_orig = [], []
    for i, line_x in enumerate(ifp_orig):

        if i > len(src_list) - 1:
            break
        x_orig.append(line_x.rstrip())

    ifp_orig.close()

    print len(tgt_list), len(src_list)

    src_list_raw = reverse_batch(src_list_raw, batch_idx)
    attn_list = reverse_batch(attn_list, batch_idx)
    src_list = reverse_batch(src_list, batch_idx)
    tgt_list = reverse_batch(tgt_list, batch_idx)

    for a_ls, x_ls,  x_ls_r, x_o, y_ls in zip(attn_list, src_list, src_list_raw, x_orig, tgt_list):
        assert len(x_ls) == len(x_ls_r)

        most_used_idxs_map = get_most_used(a_ls, y_ls)
        doc = ' '.join(x_ls_r)

        sentences = sent_tokenize(doc.encode('utf-8'))
        sentences_orig = sent_tokenize(x_o)
        token_idx = 0

        for sent, sent_o in zip(sentences, sentences_orig):
            total_in_sent = 0
            highlight_ls = []
            s_split = sent.split()
            s_split_orig = sent_o.split()

            if len(s_split) < len(s_split_orig):
                print '_____', sent
                print '_____', sent_o
                break

            for i, _ in enumerate(s_split):
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

            data['x'].append(sent)
            data['x_o'].append(sent_o)
            data['y'].append(single_y)

            print 'S', sent
            print 'O', sent_o

            len_ls.append(len(data['x'][-1].split()))
        print '\n'
    json.dump(data, ofp_json)
    ofp_json.close()

    print num_pos/float(len(data['y'])), len(data['y']), len(data['x'])
    print np.median(len_ls), np.mean(len_ls)


def reverse_batch(x, batch_idx_ls, batch_size=128):
    ret_arr, batched_arr, cur_batch = [], [], []
    batch_idx = 0

    for item_x, item_b in zip(x, batch_idx_ls):

        if batch_idx < batch_size:
            cur_batch.append((item_x, item_b))
            batch_idx += 1
        else:
            batched_arr.append(cur_batch)

            cur_batch = []

            cur_batch.append((item_x, item_b))
            batch_idx = 1

    for batch in batched_arr:

        batch.sort(key=text_sort_key)

        for item, i in batch:
            ret_arr.append(item)

    return ret_arr


def get_most_used(a_ls, y_ls):
    most_used = dict()

    for i, sub_ls_a, y in zip(range(len(a_ls)), a_ls, y_ls):
        if y == '</s>' or y == '<blank>':
            continue
        most_used[np.argmax(sub_ls_a)] = i

    return most_used


def text_sort_key(ex):
    return ex[1]


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


create_labels()