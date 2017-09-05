import multiprocessing

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


with open('/projects/btl/zxue/tasrkleat-TCGA-results/firebrowse-data/bulk_rsem_TPM/rsem-genes.txt') as inf:
    RSEM_GENES = [_.strip() for _ in inf.readlines()]


def create_label(row):
    disease, sstype = row.disease, row.sstype
    return (disease + '-' + sstype)\
        .replace('tumour', 'T')\
        .replace('normal', 'N')


def create_label_with_merging(row):
    disease, sstype = row.disease, row.sstype
    if disease in ['KICH', 'KIRC', 'KIRP'] and sstype == 'normal':
                return 'KI3-N'

    if disease in ['KICH', 'KIRC', 'KIRP'] and sstype == 'tumor':
            return 'KI3-T'

    if disease in ['LUAD', 'LUSC'] and sstype == 'normal':
            return 'LU2-N'

    if disease in ['COAD', 'READ'] and sstype == 'tumour':
            return 'CORE-T'

    if disease in ['UCEC', 'UCS'] and sstype == "tumour":
            return 'UC2-T'

    if disease in ['ESCA', 'STAD'] and sstype == "tumour":
            return 'ESST-T'

    if disease in ['BLCA', 'PRAD', 'HNSC']:
            return disease + '-NT'

    return (disease + '-' + sstype)\
        .replace('tumour', 'T')\
        .replace('normal', 'N')


def filter_by_label_group(df_trva, df_test, lg):
    df_trva = df_trva.query('label in {0}'.format(lg)).reset_index(drop=True)
    df_test = df_test.query('label in {0}'.format(lg)).reset_index(drop=True)
    return df_trva, df_test


def preprocess(df_trva, df_test, label_group):
    lbenc = LabelEncoder()
    lbenc.fit(label_group)

    df_trva['label_enc'] = lbenc.transform(df_trva.label)
    df_test['label_enc'] = lbenc.transform(df_test.label)

    classes = np.sort(df_trva.label.unique())
    df_trva['label_bin'] = label_binarize(
        df_trva.label, classes=classes).tolist()
    df_test['label_bin'] = label_binarize(
        df_test.label, classes=classes).tolist()

    sscaler = StandardScaler()
    sscaler.fit(df_trva[RSEM_GENES])
    df_trva[RSEM_GENES] = sscaler.transform(df_trva[RSEM_GENES])
    df_test[RSEM_GENES] = sscaler.transform(df_test[RSEM_GENES])
    return df_trva, df_test


def get_num_genes(clf):
    df_coefs = pd.DataFrame(clf.coef_, columns=RSEM_GENES)
    df_coefs = df_coefs.abs().sum().sort_values(ascending=False)
    df_coefs = df_coefs[df_coefs > 0]
    return df_coefs.shape[0]


def cv_fit(X_tra, y_tra, X_val, y_val, c_val):
    clf = LogisticRegression(penalty='l1', C=c_val, tol=1e-12)
    clf.fit(X_tra, y_tra)

    pred_probs = clf.predict_proba(X_val)

    logloss = metrics.log_loss(y_val, pred_probs)
    num_classes = np.unique(y_tra).shape[0]
    if num_classes == 2:
        auroc = metrics.roc_auc_score(y_val, pred_probs[:, 1])
    else:
        bin_labels = label_binarize(y_val, classes=np.arange(num_classes))
        auroc = metrics.roc_auc_score(bin_labels, pred_probs, average='macro')
    num_genes = get_num_genes(clf)

    return c_val, logloss, auroc, num_genes


def cv_fit_wrapper(args):
    return cv_fit(*args)


def gen_cv_args(df_trva, Cs=None):
    if Cs is None:
        Cs = 10 ** np.linspace(-2, 4, 50)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    Xs = df_trva[RSEM_GENES].values
    ys = df_trva.label_enc.values
    cv_args = []

    for c_val in Cs:
        for k, (idx_tra, idx_val) in enumerate(skf.split(Xs, ys)):
            X_tra = Xs[idx_tra]
            y_tra = ys[idx_tra]
            X_val = Xs[idx_val]
            y_val = ys[idx_val]

            cv_args.append([X_tra, y_tra, X_val, y_val, c_val])
    return cv_args


def cv(cv_args, num_cpus=50):
    with multiprocessing.Pool(num_cpus) as p:
        res = p.map(cv_fit_wrapper, cv_args)
    return res


def calc_cv_num_genes(df_trva, c_val):
    clf = LogisticRegression(C=c_val, penalty='l1', tol=1e-10)
    clf.fit(df_trva[RSEM_GENES].values, df_trva.label_enc.values)
    return get_num_genes(clf)


def calc_cv_num_genes_wrapper(args):
    return calc_cv_num_genes(*args)
