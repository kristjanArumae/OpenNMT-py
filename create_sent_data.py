import json
from nltk.tokenize import sent_tokenize
import numpy as np
import io
import os

import sys
import copy

reload(sys)
sys.setdefaultencoding('utf8')


def create_heatmap2(src_list, tgt_list, attn_list, data_split='train', output_to_html=-1):

    if output_to_html < 0:
        return

    ofp = open(data_split + '_readable_heatmap.html', 'w+')
    ofp.write('<html>')
    ofp.write('<script>\n'
              'function hl(id){\n'
              'document.getElementById(id).style.backgroundColor = "lightgreen";}\n'
              'function un_hl(id){\n'
              'document.getElementById(id).style.backgroundColor = "";}\n'
              '</script>\n')
    ofp.write('<body>')

    doc_id = 0

    assert len(tgt_list) == len(src_list)
    for k, (a_ls, x_ls, y_ls) in enumerate(zip(attn_list, src_list, tgt_list)):
        if k == output_to_html:
            break

        ofp.write('<p>')

        most_used_idxs_map = get_most_used(a_ls, y_ls)

        doc = ' '.join(x_ls).encode('utf-8')
        sentences = sent_tokenize(doc)
        token_idx = 0

        for sent in sentences:

            total_in_sent = 0
            token_copy = token_idx
            highlight_ls = []

            for i, _ in enumerate(sent.split()):
                if token_copy in most_used_idxs_map:
                    total_in_sent += 1
                    highlight_ls.append(1)
                else:
                    highlight_ls.append(0)

                token_copy += 1

            for word, hl, i in zip(sent.split(), highlight_ls, xrange(len(highlight_ls))):
                if word == '<unk>':
                    word = 'UNK'
                if word == '<t>':
                    word = 'T_BEGIN'
                if word == '</t>':
                    word = 'T_END'
                if word == '<blank>':
                    word = 'PADDING'

                if token_idx in most_used_idxs_map:
                    target_id = str(doc_id) + '_' + str(most_used_idxs_map[token_idx])
                    ofp.write('<span class="tag" style="background-color: rgba(255, 0, 0, 0.7);" onmouseover=hl("' + target_id + '") onmouseout=un_hl("' + target_id + '")>' + word + ' </span > ')
                else:
                    ofp.write('<span class="tag">' + word + ' </span > ')

                token_idx += 1
            ofp.write('</br>')

        ofp.write('</br>')

        for i, item in enumerate(y_ls):
            if item == '<unk>':
                item = 'UNK'
            if item == '<t>':
                item = 'T_BEGIN'
            if item == '</s>':
                item = 'S_END'
            if item == '</t>':
                item = 'T_END'
            if item == '<blank>':
                continue

            target_id = str(doc_id) + '_' + str(i)
            ofp.write('<span id="'+target_id+'">' + item + ' </span > ')

        ofp.write('</p>')
        doc_id += 1


    ofp.write('<html><body>')
    ofp.close()

def create_labels(data_split='train', output_to_html=-1, num_attn_files=1):
    ifp_v = open('vocab.json', 'rb')

    vocab_map = json.load(ifp_v)

    src_list = []
    tgt_list = []
    tgt_list_raw = []
    attn_list = []
    src_list_raw = []
    len_ls = []
    batch_idx = []

    output_path_model = 'data.nosync/' + data_split + '/model/'
    output_path_system_sent = 'data.nosync/' + data_split + '/system_sent/'
    output_path_system_segm = 'data.nosync/' + data_split + '/system_segm/'

    if not os.path.exists(output_path_model):
        os.mkdir(output_path_model)
    if not os.path.exists(output_path_system_sent):
        os.mkdir(output_path_system_sent)
    if not os.path.exists(output_path_system_segm):
        os.mkdir(output_path_system_segm)

    num_pos = 0

    ofp_json = open('data.nosync/' + data_split + '/cnndm.json', 'w+')
    ofp_html = None

    if output_to_html > 0:
        ofp_html = open('data.nosync/' + data_split + '/cnndm.html', 'w+')

    data = {'x': [], 'x_o': [], 'y': [], 's_id': [], 'batch_id': [], 'rouge': dict()}

    for i in xrange(num_attn_files):

        ifp_model = open('stanford_attn_' + data_split + str(i), 'rb')
        ifp_data = np.load(ifp_model)

        for j, sample in enumerate(ifp_data):
            src_ls_sample = sample[0]
            tgt_ls_sample = sample[1]
            attn_ls_sample = sample[2]
            batch_idx_sample = sample[3]

            src_ls_raw = [vocab_map[str(x)] for x in src_ls_sample]
            tgt_ls_raw = [vocab_map[str(y)] for y in tgt_ls_sample][1:]

            src_list.append(src_ls_sample)
            tgt_list.append(tgt_ls_sample)
            attn_list.append(attn_ls_sample)
            src_list_raw.append(src_ls_raw)
            tgt_list_raw.append(tgt_ls_raw)
            batch_idx.append(batch_idx_sample)

        ifp_model.close()

    create_heatmap2(src_list_raw, tgt_list_raw, attn_list, data_split, output_to_html)


    ifp_hl = io.open("data.nosync/" + data_split + ".txt.tgt", mode="r", encoding="utf-8")
    ifp_orig = io.open("data.nosync/" + data_split + ".txt.src", mode="r", encoding="utf-8")

    x_orig, y_orig = dict(), dict()

    for i, line_x in enumerate(ifp_orig):
        x_orig[i] = line_x.rstrip()

    for i, line_y in enumerate(ifp_hl):
        y_orig[i] = line_y.rstrip()

    ifp_orig.close()
    ifp_hl.close()

    total_d = len(src_list)
    rouge_counter = 0
    total_unused = 0

    num_sent_used, num_w_used, compression_ratio = [], [], []
    used_art = set()

    for k, (a_ls, x_ls,  x_ls_r, y_ls, b_id) in enumerate(zip(attn_list, src_list, src_list_raw, tgt_list, batch_idx)):
        assert len(x_ls) == len(x_ls_r)

        if (k + 1) % 50000 == 0:
            print 'at doc', k, '/', total_d

        assert b_id not in used_art

        try:
            x_o = x_orig[b_id]
            y_o = y_orig[b_id]
        except KeyError:
            print 'KeyError', k
            continue

        used_art.add(b_id)

        most_used_idxs_map = get_most_used(a_ls, y_ls)
        doc = ' '.join(x_ls_r)

        try:
            sentences = sent_tokenize(doc.encode('utf-8'))
        except UnicodeDecodeError:
            total_unused += 1
            continue

        if k < output_to_html:
            ofp_html.write('<p>##' + str(k) + '##</br>')

        if data_split != 'train':
            ofp_mod = open(output_path_model + 'd_' + str(rouge_counter).zfill(6) + '.txt', 'w+')
            ofp_sys_sent = open(output_path_system_sent + 'sum.' + str(rouge_counter).zfill(6) + '.txt', 'w+')
            ofp_sys_segm = open(output_path_system_segm + 'sum.' + str(rouge_counter).zfill(6) + '.txt', 'w+')

            ofp_mod.write(y_o.encode('utf-8').replace('<t>', '').replace('</t>', ''))
            ofp_mod.close()

        sentences_orig = sent_tokenize(x_o)
        token_idx = 0

        num_used, num_used_w = 0, 0

        data['rouge'][b_id] = y_o.encode('utf-8').replace('<t>', '').replace('</t>', '')

        for send_idx, (sent, sent_o) in enumerate(zip(sentences, sentences_orig)):
            total_in_sent = 0
            highlight_ls_pre_combined = []
            s_split = sent.split()
            s_split_orig = sent_o.split()

            if len(s_split) < len(s_split_orig):
                break

            for i, _ in enumerate(s_split):
                if token_idx in most_used_idxs_map:
                    total_in_sent += 1
                    highlight_ls_pre_combined.append(1)
                else:
                    highlight_ls_pre_combined.append(0)

                token_idx += 1

            highlight_ls, longest_span = combine_chunks(highlight_ls_pre_combined)

            sentence_label = total_in_sent >= 3 and longest_span[1] - longest_span[0] + 1 > 5

            single_y = [int(sentence_label)]
            sentence_length = len(s_split_orig)
            if k < output_to_html:
                onmt_str, orig_str, all_attn_str = [], [], []

                for j, (w, w_o, hl, hl_p) in  enumerate(zip(s_split, s_split_orig, highlight_ls, highlight_ls_pre_combined)):
                    if w == '<unk>':
                        w = 'UNK'
                    if w == '<t>':
                        w = 'T_BEGIN'
                    if w == '</t>':
                        w = 'T_END'
                    if w == '<blank>':
                        w = 'PADDING'

                    if hl > 0 and longest_span[0] <= j <= longest_span[1] and sentence_label:
                        onmt_str.append('<span class="tag" style="background-color: rgba(255, 0, 0, 0.7);">' + w + ' </span >')
                        orig_str.append('<span class="tag" style="background-color: rgba(255, 0, 0, 0.7);">' + w_o + ' </span >')
                    else:
                        onmt_str.append('<span class="tag">' + w + ' </span >')
                        orig_str.append('<span class="tag">' + w_o + ' </span >')

                    if hl_p > 0:
                        all_attn_str.append('<span class="tag" style="background-color: rgba(0, 255, 0, 0.7);">' + w_o + ' </span >')
                    else:
                        all_attn_str.append('<span class="tag">' + w_o + ' </span >')


                ofp_html.write(' '.join(all_attn_str).encode('utf-8'))
                ofp_html.write('</br>')

                ofp_html.write(' '.join(onmt_str).encode('utf-8'))
                ofp_html.write('</br>')

                ofp_html.write(' '.join(orig_str).encode('utf-8'))
                ofp_html.write('</br></br>')

            if sentence_label:
                single_y.append(longest_span[0])
                single_y.append(longest_span[1])
                span_len = longest_span[1] - longest_span[0]

                compression_ratio.append(span_len / float(sentence_length))

                num_used_w += span_len

                if data_split != 'train':
                    ofp_sys_sent.write(sent_o.encode('utf-8') + ' ')
                    ofp_sys_segm.write(' '.join(s_split_orig[longest_span[0]:longest_span[1]]).encode('utf-8') + ' ')

                num_pos += 1

                num_used += 1
            else:
                single_y.extend([-1, -1])

            data['x'].append(sent)
            data['x_o'].append(sent_o)
            data['y'].append(single_y)
            data['s_id'].append(send_idx)
            data['batch_id'].append(b_id)

            len_ls.append(len(data['x'][-1].split()))

        if num_used == 0:
            # print k
            total_unused += 1

        if data_split != 'train':
            ofp_sys_segm.close()
            ofp_sys_sent.close()

            rouge_counter += 1

        if ofp_html is not None:
            ofp_html.write('<p>')

        num_sent_used.append(num_used)
        num_w_used.append(num_used_w)


    json.dump(data, ofp_json)
    ofp_json.close()

    if ofp_html is not None:
        ofp_html.close()

    print data_split
    print num_pos/float(len(data['y']) + 1.0), len(data['y']), len(data['x']), total_unused
    print np.median(len_ls), np.mean(len_ls)
    print 'sent used', np.mean(num_sent_used), 'w used', np.mean(num_w_used)
    print 'comp ratio', np.mean(compression_ratio), '\n'


def reverse_batch(x, batch_idx_ls):
    ret_arr, batched_arr = [], []

    for item_x, item_b in zip(x, batch_idx_ls):
        batched_arr.append((item_x, item_b))

    batched_arr.sort(key=text_sort_key)

    prev_i = -1
    for item, i in batched_arr:
        ret_arr.append(item)

        assert i - prev_i == 0
        prev_i = i

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


def combine_chunks(highlight_ls_):
    longest_span = (None, None)
    distance_from_hl = 0
    is_begin = True

    highlight_ls = copy.copy(highlight_ls_)

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

    cur_len = 0

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
        longest_span = (len(highlight_ls)  - cur_len, len(highlight_ls) - 1)
    elif cur_len > 0 and cur_len > longest_span[1] - longest_span[0] + 1:
        longest_span = (len(highlight_ls)  - cur_len, len(highlight_ls) - 1)

    return highlight_ls, longest_span


create_labels(data_split='train', output_to_html=5000, num_attn_files=5)
create_labels(data_split='valid', output_to_html=5000, num_attn_files=1)
create_labels(data_split='test', output_to_html=5000, num_attn_files=1)