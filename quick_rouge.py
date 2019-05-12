from pyrouge import Rouge155
import tempfile


def run_rouge(data_split='valid', by=False, s='system_segm'):
    tempfile.tempdir = '/scratch/'

    rouge_fname = 'data.nosync/' + data_split + '/model/'

    if by:
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.model_dir = rouge_fname
    r.system_dir = 'data.nosync/' + data_split + '/' + s + '/'

    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'd_#ID#.txt'

    fname = data_split + '_' + s + '_rouge.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

# output_path_model = 'data.nosync/' + data_split + '/model/'
# output_path_system_sent = 'data.nosync/' + data_split + '/system_sent/'
# output_path_system_segm = 'data.nosync/' + data_split + '/system_segm/'

run_rouge()
run_rouge(s='system_sent')
