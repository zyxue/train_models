import multiprocessing
import logging

import pandas as pd
import numpy as np

import cv

logging.basicConfig(
    level=logging.DEBUG, format='%(asctime)s|%(levelname)s|%(message)s')


LABEL_GROUPS = [
    ['LUAD-N', 'LUSC-N'],
    ['BLCA-N', 'BLCA-T'],
    ['HNSC-N', 'HNSC-T'],
    ['UCEC-T', 'UCS-T'],
    ['ESCA-T', 'STAD-T'],
    ['PRAD-N', 'PRAD-T'],
    ['COAD-T', 'READ-T'],

    ['KICH-N', 'KIRC-N', 'KIRP-N'],
    ['KICH-T', 'KIRC-T', 'KIRP-T']
]


DF_TRVA = pd.read_pickle('../data/transformed-data/train_test_split/80-20/train.pkl')
DF_TEST = pd.read_pickle('../data/transformed-data/train_test_split/80-20/test.pkl')

for lg in LABEL_GROUPS:
    print('working on {0}'.format(lg))

    df_trva, df_test = cv.filter_by_label_group(DF_TRVA, DF_TEST, lg)
    df_trva, df_test = cv.preprocess(df_trva, df_test, lg)

    Cs = 10 ** np.arange(-2, 4, 0.1)
    logging.info('working on {0} C values'.format(len(Cs)))
    cv_args = cv.gen_cv_args(df_trva, Cs)
    with multiprocessing.Pool(60) as p:
        cv_res = p.map(cv.cv_fit_wrapper, cv_args)

    df_cv = pd.DataFrame(
        cv_res, columns=['c_val', 'logloss', 'auroc', 'num_genes'])

    df_cv.to_csv(
        './second-stage-models/{0}-cv.csv'.format('-'.join(lg)), index=False)

    with multiprocessing.Pool(50) as p:
        num_genes_trva = p.map(cv.calc_cv_num_genes_wrapper,
                          [(df_trva, _) for _ in Cs])

    df_num_genes_trva = pd.DataFrame(
        np.array([Cs, num_genes_trva]).T, columns=['c_val', 'num_genes'])
    df_num_genes_trva.to_csv(
        './second-stage-models/{0}-num-genes-trva.csv'.format('-'.join(lg)), index=False)
