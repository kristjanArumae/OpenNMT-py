from pyrouge import Rouge155
import tempfile


def run_rouge(data_split='train', by=False, s='small_sys_sent'):
    tempfile.tempdir = '/scratch/'

    rouge_fname = 'data.nosync/' + data_split + '/small_model/'

    if by:
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.model_dir = rouge_fname
    r.system_dir = 'data.nosync/' + data_split + '/' + s + '/'

    r.system_filename_pattern = 's_(\d+).txt'
    r.model_filename_pattern = 'm_#ID#.txt'

    fname = s + '_rouge.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

# output_path_model = 'data.nosync/' + data_split + '/model/'
# output_path_system_sent = 'data.nosync/' + data_split + '/system_sent/'
# output_path_system_segm = 'data.nosync/' + data_split + '/system_segm/'

run_rouge()
run_rouge(s='small_sys_segs')
