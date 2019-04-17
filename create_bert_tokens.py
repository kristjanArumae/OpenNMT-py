import json
from pytorch_pretrained_bert import BertTokenizer


def tokenize_data(data_split='train', max_len=30, output_to_html=-1, small_subset=-1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ifp = open('data.nosync/' + data_split + '/cnndm.json', 'rb')
    data = json.load(ifp)

    ifp.close()

    ofp_html = None

    if output_to_html > 0:
        ofp_html = open('data.nosync/' + data_split + '/cnndm_bert.html', 'w+')

    sent_tokenized_ls = []
    updated_labels_ls = []

    for i, (sent, labels) in enumerate(zip(data['x_o'], data['y'])):

        if i == small_subset:
            break

        word_ls = ['[CLS]'] + sent.split()
        label_begin = labels[1] + 1 if labels[1] != -1 else -1
        label_end = labels[2] + 1 if labels[2] != -1 else -1
        location = 0

        sent_tokenized_as_idx = []
        sent_tokenized_as_tok = []
        moving_idx = 0

        for _, word in enumerate(word_ls):
            if moving_idx >= label_begin and location == 0:
                location += 1
            elif moving_idx > label_end and location == 1:
                location += 1

            tok = tokenizer.tokenize(word)
            x = tokenizer.convert_tokens_to_ids(tok)
            len_x = len(x)

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

        if labels[0] > 0:
            updated_labels_ls.append([1, label_begin, label_end])
        else:
            updated_labels_ls.append(labels)
        sent_tokenized_ls.append([sent_tokenized_as_idx, sent_tokenized_as_tok])

    ofp = open('data.nosync/' + data_split + '/cnndm_labeled_tokenized.json', 'w+')

    updated_data = dict()
    updated_data['x'] = sent_tokenized_ls
    updated_data['y'] = updated_labels_ls

    print(len(updated_labels_ls))

    json.dump(updated_data, ofp)

    ofp.close()

    if output_to_html > 0:
        ofp_html.close()


tokenize_data(output_to_html=100000)