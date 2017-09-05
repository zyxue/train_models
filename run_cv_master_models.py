import sys
import multiprocessing

import pandas as pd
import numpy as np

import cv


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
    df_trva['label'] = df_trva.apply(create_label, axis=1)
    df_test['label'] = df_test.apply(create_label, axis=1)
    
elif master_model_name == "nt":
    diseases_with_sufficient_normal = [
        'BRCA', 'COAD', 'KICH',
        'KIRC', 'KIRP', 'LIHC',
        'LUAD', 'LUSC', 'THCA',
        'HNSC', 'STAD', 'UCEC',
        'BLCA', 'PRAD'
    ]

    df_trva = DF_TRVA.query('diseases in {0}'.format(diseases_with_sufficient_normal))
    df_test = DF_TEST.query('diseases in {0}'.format(diseases_with_sufficient_normal))

elif master_model_name == 'all-t':
    lg = [_ for _ in DF_TRVA.label.unique() if _.endswith('-T')]
    df_trva, df_test = cv.filter_by_label_group(DF_TRVA, DF_TEST, lg)
    df_trva, df_test = cv.preprocess(df_trva, df_test, lg)


Cs = 10 ** np.linspace(-2, 4, 50)
cv_args = cv.gen_cv_args(df_trva, Cs)
with multiprocessing.Pool(50) as p:
    cv_res = p.map(cv.cv_fit_wrapper, cv_args)

df_cv = pd.DataFrame(
    cv_res, columns=['c_val', 'logloss', 'auroc', 'num_genes'])

with multiprocessing.Pool(50) as p:
    num_genes = p.map(cv.calc_cv_num_genes_wrapper,
                      [(df_trva, _) for _ in Cs])

df_cv.to_csv(
    './master-models/{0}-cv.csv'.format('-'.join(master_model_name)), index=False)
