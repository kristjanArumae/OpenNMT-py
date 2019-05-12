import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def create_metric_figure(fname, loss_ls, loss_ls_s, loss_ls_qa, loss_valid_ls, qa_f1, sent_f1, cur_used_ls_mean, total_used, total_s, mean_seg_len):
    plt.plot([i for i in range(len(loss_ls))], loss_ls, '-', label="loss", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_s, '-', label="sent", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_ls_qa, '-', label="qa", linewidth=1)
    plt.plot([i for i in range(len(loss_ls))], loss_valid_ls, '-', label="valid", linewidth=1)

    plt.legend(loc='best')
    plt.savefig(fname + '_loss.png', dpi=400)

    plt.clf()

    plt.plot([i for i in range(len(qa_f1))], qa_f1, '-', label="qa f1", linewidth=1)
    plt.plot([i for i in range(len(sent_f1))], sent_f1, '-', label="sent f1", linewidth=1)

    plt.legend(loc='best')
    plt.savefig(fname + '_val.png', dpi=400)

    print('\n\n\nSent used:', total_used, '/', total_s, total_used / float(total_s))
    print('Avg len (sent)', cur_used_ls_mean)
    print('avg seg len', mean_seg_len)


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


def create_valid_rouge(rouge_dict, x_for_rouge, eval_sys_sent, eval_sys_start, eval_sys_end, gt_sent, gt_start, gt_end,
                       batch_ids, align_ls, rouge_sys_sent_path, rouge_sys_segs_path, ofp_fname):

    ofp_rouge_sent = None
    ofp_rouge_segm = None

    cur_batch = -1

    used_set = set()

    total_s = 0
    total_used = 0
    cur_used = 0
    cur_used_ls = []

    uesd_seg_len = []

    ofp_readable = open('data.nosync/' + ofp_fname + '.html', 'w+')

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


def create_output_name(args):
    return 'batch_' + str(args.batch_size) + \
           '_e_' + str(args.epochs) + \
           '_m_len_' + str(args.sent_len) + \
           '_bal_' + str(args.balance) + \
           '_pre_' + str(args.bert_model) + \
           '_lr_' + str(args.lr) + \
           '_drop_' + str(args.dropout).replace('.', '_') + \
           '_op_' + str(args.optim) + \
           '_weight_' + str(args.weights) + \
           '_dpos_' + str(args.use_positional) + \
           '_lim_' + str(args.unchanged_limit)



