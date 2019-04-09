import json
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


def tokenize_data(max_len=30):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ifp = open('data.nosync/train/cnndm.json', 'rb')
    data = json.load(ifp)

    ifp.close()

    sent_tokenized_ls = []
    updated_labels_ls = []

    for i, (sent, labels) in enumerate(zip(data['x'], data['y'])):

        word_ls = sent.split()
        label_begin = labels[1]
        label_end = labels[2]
        location = 0

        sent_tokenized_as_idx = []
        sent_tokenized_as_tok = []

        for j, word in enumerate(word_ls):
            if j >= label_begin and location == 0:
                location += 1
            elif j > label_end and location == 1:
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

        if labels[0] > 0:
            updated_labels_ls.append([1, label_begin, label_end])
        else:
            updated_labels_ls.append(labels)
        sent_tokenized_ls.append([sent_tokenized_as_idx, sent_tokenized_as_tok])

    ofp = open('data.nosync/train/cnndm_labeled_tokenized.json', 'w+')

    updated_data = dict()
    updated_data['x'] = sent_tokenized_ls
    updated_data['y'] = updated_labels_ls

    print(len(updated_labels_ls))

    json.dump(updated_data, ofp)

    ofp.close()


tokenize_data()