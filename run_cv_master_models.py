import sys
import multiprocessing
import logging

import pandas as pd
import numpy as np


import cv


logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')


DF_TRVA = pd.read_pickle('../data/transformed-data/train_test_split/80-20/train.pkl')
DF_TEST = pd.read_pickle('../data/transformed-data/train_test_split/80-20/test.pkl')

master_model_name = sys.argv[1]

print('working on {0}'.format(master_model_name))

if master_model_name == 'all':
    df_trva = DF_TRVA
    df_test = DF_TEST

elif master_model_name == 'all-with-class-merge':
    df_trva = DF_TRVA
    df_test = DF_TEST
    df_trva['label'] = df_trva.apply(cv.create_label_with_merging, axis=1)
    df_test['label'] = df_test.apply(cv.create_label_with_merging, axis=1)

elif master_model_name == "nt":
    dises_with_suff_normal = [
        'BRCA', 'COAD', 'KICH',
        'KIRC', 'KIRP', 'LIHC',
        'LUAD', 'LUSC', 'THCA',
        'HNSC', 'STAD', 'UCEC',
        'BLCA', 'PRAD'
    ]

    df_trva = DF_TRVA.query(
        'disease in {0}'.format(dises_with_suff_normal)).copy()
    df_test = DF_TEST.query(
        'disease in {0}'.format(dises_with_suff_normal)).copy()

elif master_model_name == 'all-t':
    lg = [_ for _ in DF_TRVA.label.unique() if _.endswith('-T')]
    df_trva, df_test = cv.filter_by_label_group(DF_TRVA, DF_TEST, lg)

    # merge hard-to-classify classes
    df_trva['label'] = df_trva.apply(cv.create_label_with_merging, axis=1)
    df_test['label'] = df_test.apply(cv.create_label_with_merging, axis=1)


logging.info('preprocessing...')
lg = df_trva.label.unique()
df_trva, df_test = cv.preprocess(df_trva, df_test, lg)


Cs = 10 ** np.arange(-2, 2, 0.1)
logging.info('working on {0} C values'.format(len(Cs)))

logging.info('start grid search')

grid_search_result = cv.cv_master(df_trva, Cs)
df_cv = cv.generate_df_cv_master(grid_search_result)
df_cv.to_csv(
    './master-models/{0}-cv.csv'.format(master_model_name), index=False)


# logging.info('refit to obtain gene number counts')
# with multiprocessing.Pool(40) as p:
#     num_genes_trva = p.map(cv.calc_cv_num_genes_wrapper,
#                            [(df_trva, _) for _ in Cs])

# df_num_genes_trva = pd.DataFrame(
#     np.array([Cs, num_genes_trva]).T, columns=['c_val', 'num_genes'])
# df_num_genes_trva.to_csv(
#     './master-models-models/{0}-num-genes-trva.csv'.format('-'.join(master_model_name)), index=False)
