import bo
import torch
import random
import argparse
import numpy as np
import pandas as pd
import numpy as np
from utils import transform_forward, transform_backward
import os

import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor


from preprocessor import Preprocessor  # Adjust import path as needed

# AutoGluon Configuration
AG_ARGS_FIT = {
    "ag.max_memory_usage_ratio": 0.3,
    'num_gpus': 1,
    'num_cpus': min(10, os.cpu_count() if os.cpu_count() else 4)
}

STABLE_MODELS = [
    "GBM", "CAT", "XGB", "RF", "XT", "KNN", "LR", "NN_TORCH", "FASTAI",
    "NN_MXNET", "TABPFN", "DUMMY", "NB"
]

# Pipeline configurations (same as before)
PIPELINE_CONFIGS = [
    {'name': 'baseline', 'imputation': 'none', 'scaling': 'none', 'encoding': 'none', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'simple_preprocess', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'robust_preprocess', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'},
    {'name': 'feature_selection', 'imputation': 'median', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'k_best', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'dimension_reduction', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
    {'name': 'conservative', 'imputation': 'median', 'scaling': 'minmax', 'encoding': 'onehot', 
     'feature_selection': 'variance_threshold', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'aggressive', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'},
    {'name': 'knn_impute_pca', 'imputation': 'knn', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
    {'name': 'mutual_info_zscore', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 
     'feature_selection': 'mutual_info', 'outlier_removal': 'zscore', 'dimensionality_reduction': 'none'},
    {'name': 'constant_maxabs_iforest', 'imputation': 'constant', 'scaling': 'maxabs', 'encoding': 'onehot', 
     'feature_selection': 'variance_threshold', 'outlier_removal': 'isolation_forest', 'dimensionality_reduction': 'none'},
    {'name': 'mean_minmax_lof_svd', 'imputation': 'mean', 'scaling': 'minmax', 'encoding': 'onehot', 
     'feature_selection': 'k_best', 'outlier_removal': 'lof', 'dimensionality_reduction': 'svd'},
    {'name': 'mostfreq_standard_iqr', 'imputation': 'most_frequent', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'}
]


def evaluate_single_pipeline_autogluon(X, y, pipeline_config, test_size=0.3, random_state=42):
    """
    Evaluate a single pipeline using AutoGluon
    Returns accuracy score (0-100 scale)
    """
    try:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply preprocessing pipeline
        preprocessor = Preprocessor(pipeline_config)
        preprocessor.fit(X_train, y_train)
        
        X_train_proc, y_train_proc = preprocessor.transform(X_train, y_train)
        X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test)
        
        # Check if preprocessing was successful
        if len(X_train_proc) < 10 or len(X_test_proc) < 5:
            return np.nan
        
        # Prepare data for AutoGluon
        train_data = X_train_proc.copy()
        train_data['target'] = y_train_proc.values
        
        # Train AutoGluon
        with tempfile.TemporaryDirectory() as temp_dir:
            problem_type = 'binary' if y_train_proc.nunique() <= 2 else 'multiclass'
            
            predictor = TabularPredictor(
                label='target',
                path=temp_dir,
                problem_type=problem_type,
                eval_metric='accuracy',
                verbosity=0
            )
            
            predictor.fit(
                train_data,
                time_limit=600,
                presets='medium_quality',
                included_model_types=STABLE_MODELS,
                hyperparameter_tune_kwargs=None,
                feature_generator=None,
                ag_args_fit=AG_ARGS_FIT,
                raise_on_no_models_fitted=False
            )
            
            # Predict and evaluate
            preds = predictor.predict(X_test_proc)
            accuracy = accuracy_score(y_test_proc, preds) * 100
            
            return accuracy
            
    except Exception as e:
        print(f"      Error evaluating pipeline {pipeline_config['name']}: {e}")
        return np.nan


def evaluate_all_pipelines_on_dataset(X, y, pipeline_indices=None):
    """
    Evaluate all (or selected) pipelines on a single dataset using AutoGluon
    
    Args:
        X: Dataset features
        y: Dataset labels
        pipeline_indices: Optional list of pipeline indices to evaluate (None = all)
        
    Returns:
        accuracies: Array of accuracy scores for each pipeline (NaN for failed ones)
    """
    if pipeline_indices is None:
        configs_to_eval = PIPELINE_CONFIGS
    else:
        configs_to_eval = [PIPELINE_CONFIGS[i] for i in pipeline_indices]
    
    accuracies = []
    
    for config in configs_to_eval:
        print(f"    Evaluating: {config['name']:30}", end=" ")
        acc = evaluate_single_pipeline_autogluon(X, y, config)
        accuracies.append(acc)
        
        if not np.isnan(acc):
            print(f"→ {acc:.2f}%")
        else:
            print("→ FAILED")
    
    return np.array(accuracies)


def evaluate_recommended_pipelines(X, y, recommended_indices, n_eval=None):
    """
    Evaluate the top-N recommended pipelines and compare to baseline/best
    
    Args:
        X: Dataset features
        y: Dataset labels  
        recommended_indices: Ordered list of pipeline indices (best first)
        n_eval: Number of top recommendations to evaluate (None = all)
        
    Returns:
        results: Dict with evaluation metrics and rankings
    """
    if n_eval is None:
        n_eval = len(recommended_indices)
    
    # Evaluate recommended pipelines
    print(f"\n  Evaluating top {n_eval} recommended pipelines:")
    recommended_scores = []
    evaluated_indices = []
    
    for i, idx in enumerate(recommended_indices[:n_eval]):
        config = PIPELINE_CONFIGS[idx]
        print(f"    Rank {i+1}: {config['name']:30}", end=" ")
        acc = evaluate_single_pipeline_autogluon(X, y, config)
        recommended_scores.append(acc)
        evaluated_indices.append(idx)
        
        if not np.isnan(acc):
            print(f"→ {acc:.2f}%")
        else:
            print("→ FAILED")
    
    # Evaluate baseline (pipeline index 0)
    print(f"\n  Evaluating baseline:")
    baseline_config = PIPELINE_CONFIGS[0]
    baseline_score = evaluate_single_pipeline_autogluon(X, y, baseline_config)
    print(f"    {baseline_config['name']:30} → {baseline_score:.2f}%")
    
    # Evaluate all pipelines to find true best
    print(f"\n  Evaluating all {len(PIPELINE_CONFIGS)} pipelines to find ground truth:")
    all_scores = evaluate_all_pipelines_on_dataset(X, y)
    
    # Compute rankings with tie handling
    valid_mask = ~np.isnan(all_scores)
    rankings = np.full(len(all_scores), np.nan)
    
    if valid_mask.sum() > 0:
        from scipy.stats import rankdata
        rankings[valid_mask] = rankdata(-all_scores[valid_mask], method='average')
    
    # Find best pipeline
    best_idx = np.nanargmax(all_scores)
    best_score = all_scores[best_idx]
    best_rank = rankings[best_idx]
    
    # Get rank of first recommended pipeline
    first_rec_idx = evaluated_indices[0]
    first_rec_score = all_scores[first_rec_idx]
    first_rec_rank = rankings[first_rec_idx]
    
    # Summary
    results = {
        'all_scores': all_scores,
        'rankings': rankings,
        'recommended_indices': evaluated_indices,
        'recommended_scores': recommended_scores,
        'baseline_score': baseline_score,
        'baseline_rank': rankings[0],
        'best_pipeline_idx': best_idx,
        'best_pipeline_name': PIPELINE_CONFIGS[best_idx]['name'],
        'best_score': best_score,
        'best_rank': best_rank,
        'first_recommended_idx': first_rec_idx,
        'first_recommended_name': PIPELINE_CONFIGS[first_rec_idx]['name'],
        'first_recommended_score': first_rec_score,
        'first_recommended_rank': first_rec_rank,
        'top_k_hit': first_rec_idx == best_idx  # Did we recommend the best?
    }
    
    print(f"\n  {'='*60}")
    print(f"  RESULTS:")
    print(f"  {'='*60}")
    print(f"  Baseline:           {baseline_config['name']:30} Acc={baseline_score:.2f}% Rank={rankings[0]:.1f}")
    print(f"  Best Pipeline:      {PIPELINE_CONFIGS[best_idx]['name']:30} Acc={best_score:.2f}% Rank={best_rank:.1f}")
    print(f"  1st Recommended:    {PIPELINE_CONFIGS[first_rec_idx]['name']:30} Acc={first_rec_score:.2f}% Rank={first_rec_rank:.1f}")
    print(f"  Top-1 Hit:          {'✓ YES' if results['top_k_hit'] else '✗ NO'}")
    print(f"  {'='*60}\n")
    
    return results

'''load arguments'''
def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name', default='pmf', help='pmf, 110-classifiers, openml')
    parser.add_argument(
        '--save_path', default='default', help='the path to save result')
    parser.add_argument(
        '--warm_path', default='default', help='the path to warm starters setting')
    parser.add_argument(
        '--data_path', default='None', help='the path to save data')
    
        
    parser.add_argument(
        '--random_seed', type=int, default=0, help='for random seed')
    parser.add_argument(
        '--nan_ratio', type=float, default=0, help='for random seed')
    parser.add_argument(
        '--save_name', default='default', help='save the reuslts')
    parser.add_argument(
        '--part_name', default='0', help='part')
    parser.add_argument(
        '--model_path', default='None', help='the pmf model path')
    parser.add_argument(
        '--warm_start', default='default', help='the path for the define of warm starters')
    parser.add_argument(
        '--warm_trained', type=int, default=0, help='is the warm starter is trained')
    parser.add_argument(
        '--bo_n_init', type=int, default=5, help='the number of pipelines for warm start')
    parser.add_argument(
        '--bo_n_iters', type=int, default=200, help='the number of pipelines for search')
    parser.add_argument(
        '--is_bayes', type=int, default=1)
    parser.add_argument(
        '--is_narrow', type=int, default=0)

    args, unparsed = parser.parse_known_args()
    return args


'''set random seed'''
def setup_seed(seed):
    #setup_seed()使用后，只作用于random()函数一次，如之后再次调用random（）函数>，则需要再次调用setup_seed
     torch.manual_seed(seed) #cpu
     torch.cuda.manual_seed(seed)#当前gpu
     torch.cuda.manual_seed_all(seed) #所有gpu
     np.random.seed(seed) #numpy
     random.seed(seed) #python
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True


'''get data'''
def get_data(dataset_name='openml', pipeline_ixs=None, save_path=None, nan_ratio=0, data_path='None', random_seed=0):
    """
    returns the train/test splits of the dataset as N x D matrices and the
    train/test dataset features used for warm-starting bo as D x F matrices.
    N is the number of pipelines, D is the number of datasets (in train/test),
    and F is the number of dataset features.
    """

    fn_data = '/drive1/nammt/SoluRec/Data/preprocessed_performance.csv'.format(dataset_name)
    fn_data_feats = '/drive1/nammt/SoluRec/Data/dataset_feats.csv'.format(dataset_name)
    # fn_pipelines_feats = '../data/{}/pipelines.json'.format(dataset_name)


    pipeline_names = [
        'baseline', 'simple_preprocess', 'robust_preprocess', 'feature_selection',
        'dimension_reduction', 'conservative', 'aggressive', 'knn_impute_pca',
        'mutual_info_zscore', 'constant_maxabs_iforest', 'mean_minmax_lof_svd',
        'mostfreq_standard_iqr'
    ]

    df = pd.read_csv(fn_data, index_col=0)

    if pipeline_ixs is not None:
        df = df.iloc[pipeline_ixs]
    
    dataset_ids = [col for col in df.columns]

    try:
        dataset_ids = [int(d.replace('D_', '').replace('D', '')) for d in dataset_ids]
    except:
        pass

    Y = df.values.astype(np.float64)

    if data_path == 'None':
        train_dataset_ids = [
            22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55,
            56, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
            187, 188, 275, 276, 277, 278, 285, 300, 301, 307, 308,
            310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
            337, 338, 339, 340, 342, 343, 346, 372, 375
        ]
        
        test_dataset_ids = [
            1503, 23517, 1551, 1552, 183, 255, 545, 546, 475, 481, 
            516, 3, 6, 8, 10, 12, 14, 9, 11, 5
        ]

        ids_train = [d for d in train_dataset_ids if d in dataset_ids]

        ids_test = test_dataset_ids



        random_rank = np.random.permutation(ids_train)
        ids_train = random_rank[:int(len(random_rank)*0.8)]
        ids_val = random_rank[int(len(random_rank)*0.8):]

        ids_train = list(map(int, ids_train))
        missing_ids = [i for i in ids_train if i not in dataset_ids]
        if missing_ids:
            raise ValueError(f"The following IDs are missing from dataset_ids: {missing_ids}")

        # Create a mapping for faster lookups
        dataset_id_to_index = {id: idx for idx, id in enumerate(dataset_ids)}

        # Get indices for train, validation, and test sets
        ix_train = [dataset_id_to_index[i] for i in ids_train if i in dataset_id_to_index]
        ix_val = [dataset_id_to_index[i] for i in ids_val if i in dataset_id_to_index]
        ix_test = [dataset_id_to_index[i] for i in ids_test if i in dataset_id_to_index]

        import os
        os.makedirs(f'../result/{dataset_name}/{save_path}', exist_ok=True)
        np.save(f'..resuklt=')
        np.save(f'../result/{dataset_name}/{save_path}/ids_train.npy', np.array(ids_train))
        np.save(f'../result/{dataset_name}/{save_path}/ids_val.npy', np.array(ids_val))
        np.save(f'../result/{dataset_name}/{save_path}/ids_test.npy', np.array(ids_test))

    else:
        ids_train = np.load(f'../result/{dataset_name}/{data_path}/ids_train.npy')
        ids_val = np.load(f'../result/{dataset_name}/{data_path}/ids_val.npy')
        ids_test = np.load(f'../result/{dataset_name}/{data_path}/ids_test.npy')



    # if dataset_name == 'pmf':
    #     fn_train_ix = '../data/{}/ids_train.csv'.format(dataset_name)
    #     fn_test_ix = '../data/{}/ids_test.csv'.format(dataset_name)
    # else:
    #     fn_train_ix = None
    #     fn_test_ix = None

    # '''load performance '''
    # df = pd.read_csv(fn_data)
    # if pipeline_ixs is not None:
    #     df = df.iloc[pipeline_ixs]

    # #pipeline_ids = df['Unnamed: 0'].tolist()
    # dataset_ids = df.columns.tolist()[1:]
    # dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    # Y = df.values[:,1:].astype(np.float64)

    # if dataset_name == 'openml':
    #     Y = Y*100.

    # '''train / test'''
    # ids_train = None
    # ids_val = None
    # ids_test = None
    # if data_path=='None':
    #     if fn_train_ix is None:
    #         if dataset_name=='openml':
    #             random_rank = np.random.permutation(dataset_ids)
    #             n_train = int(len(dataset_ids)*0.8)
    #             n_test = len(dataset_ids)-n_train
    #             ids_train = []
    #             ids_test = []

    #             my_nn = 0
    #             for i in random_rank:
    #                 Yi = Y[:, dataset_ids.index(i)]

    #                 if n_test>0 and len(Yi)-np.isnan(Yi).sum()>261:
    #                     ids_test.append(i)
    #                     n_test -=1
    #                     my_nn+=1
    #                 else:
    #                     ids_train.append(i)

    #             ids_train = np.array(ids_train)
    #             ids_test = np.array(ids_test)

    #         else:
    #             random_rank = np.random.permutation(dataset_ids)
    #             ids_train = random_rank[:int(len(random_rank)*0.8)]
    #             ids_test = random_rank[int(len(random_rank)*0.8):]
    #     else:
    #         ids_train = np.loadtxt(fn_train_ix).astype(int).tolist()
    #         ids_test = np.loadtxt(fn_test_ix).astype(int).tolist()
        
    #     random_rank = np.random.permutation(ids_train)
    #     ids_train = random_rank[:int(len(random_rank)*0.8)]
    #     ids_val = random_rank[int(len(random_rank)*0.8):]

    #     np.save('../result/{}/{}/ids_train.npy'.format(dataset_name, save_path), np.array(ids_train))
    #     np.save('../result/{}/{}/ids_val.npy'.format(dataset_name, save_path), np.array(ids_val))
    #     np.save('../result/{}/{}/ids_test.npy'.format(dataset_name, save_path), np.array(ids_test))
    # else:
    #     ids_train = np.load('../result/{}/{}/ids_train.npy'.format(dataset_name, data_path))
    #     ids_val = np.load('../result/{}/{}/ids_val.npy'.format(dataset_name, data_path))
    #     ids_test = np.load('../result/{}/{}/ids_test.npy'.format(dataset_name, data_path))

    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_val = [dataset_ids.index(i) for i in ids_val]
    ix_test = [dataset_ids.index(i) for i in ids_test]


    Ytrain = Y[:, ix_train]
    Yval = Y[:, ix_val]
    Ytest = Y[:, ix_test]

    setup_seed(random_seed)
    nan_num = np.isnan(Ytrain).sum()
    total_num = Ytrain.size
    target_num = int(total_num * nan_ratio)


    while nan_num < target_num:
        a = np.random.randint(Ytrain.shape[0])
        b = np.random.randint(Ytrain.shape[1])

        while np.isnan(Ytrain[a,b]):
            a = np.random.randint(Ytrain.shape[0])
            b = np.random.randint(Ytrain.shape[1])
        
        Ytrain[a,b] = np.nan
        nan_num += 1

    '''load dataset features'''
    df = pd.read_csv(fn_data_feats)
    df.replace([np.inf,-np.inf], -1, inplace=True)
    df = df.fillna(0)
    dataset_ids = df[df.columns[0]].tolist()

    try:
        feat_dataset_ids = [int(str(d).replace('D_', '').replace('D', '')) for d in feat_dataset_ids]
    except:
        pass

    ix_train = [dataset_ids.index(i) for i in ids_train if i in feat_dataset_ids]
    ix_val = [dataset_ids.index(i) for i in ids_val if i in feat_dataset_ids]
    ix_test = [dataset_ids.index(i) for i in ids_test if i in feat_dataset_ids]

    Ftrain = df.values[ix_train, 1:]
    Fval = df.values[ix_val, 1:]
    Ftest = df.values[ix_test, 1:]

    '''add Normalize'''
    df_norm = df.copy()
    df_norm = (df_norm.iloc[:, 1:] - df_norm.iloc[:, 1:].mean()) / df_norm.iloc[:, 1:].std()

    # df_norm = df.fillna(-1)
    # df_norm = (df_norm - df_norm.mean())/df_norm.std()
    df_norm = df_norm.fillna(0)

    FtrainNorm = df_norm.values[ix_train]
    FvalNorm = df_norm.values[ix_val]
    FtestNorm = df_norm.values[ix_test]

    # FtrainNorm = df.values[ix_train, 1:]
    # FvalNorm =df.values[ix_val, 1:]
    # FtestNorm  = df.values[ix_test, 1:]

    '''Get Pipeline Feats'''
    df = pd.read_json(fn_pipelines_feats)
    df = df.fillna(-1)

    pipeline_names = df['id'].tolist()

    # if dataset_name == 'pmf':
    #     df.pop('model')
    #     df.pop('pre-processor')

    if pipeline_ixs is not None:
        df = df.iloc[pipeline_ixs]

    # for k,j in zip(df.keys(), df.dtypes):
    #     if j==object:
    #         df[k] = pd.to_numeric(df[k], errors='coerce').fillna('0').astype('int32')       

    # FPipeline = df.values
    FPipeline = np.eye(len(pipeline_names))

    return Ytrain, Yval, Ytest, Ftrain, Fval, Ftest, FtrainNorm, FvalNorm, FtestNorm, FPipeline, pipeline_names


'''warm_starter test'''
def test_warmstarter(bo_n_init, Yt, Ft, do_print=False, warm_start='l1', 
    warm_starter=None, is_narrow=0, narrow_list=None):

    # warm start
    ix_init = warm_starter.recommend(Ft)

    accs = np.zeros([Yt.shape[1], bo_n_init])
    best_accs = np.zeros([Yt.shape[1], bo_n_init])

    for ind, (ixs, y_train) in enumerate(zip(ix_init, Yt.T)):
        n_init = 0
        best_acc = 0

        for i in ixs:
            if not np.isnan(y_train[i]):
                if (is_narrow) and (i not in narrow_list):
                    continue
                accs[ind][n_init] = y_train[i]
                
                if best_acc<y_train[i]:
                    best_acc = y_train[i]
                best_accs[ind][n_init] = best_acc

                n_init += 1

                if bo_n_init==n_init:
                    break
                
    return accs, best_accs, ix_init

'''Bayesian Search'''
def bo_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None):
    
    preds = bo.BO(m.dim, m.kernel, bo.ei,
                  variance=transform_forward(m.variance))
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []

    # warm start
    ix_init = warm_starter.recommend(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            preds.add(m.X[ix], ytest[ix])
            ix_evaled.append(ix)

            ix_candidates.remove(ix)

            yb = preds.ybest
            ybest_list.append(yb)

            if do_print:
                print('Iter: %d, %g [%d], Best: %g' % (len(ybest_list), ytest[ix], ix, yb))
            
            n_init +=1
            if n_init==bo_n_init:
                break
    
    if len(ybest_list)==0:
        ix_evaled.append(-1)
        ybest_list.append(0)

    while len(ybest_list)<bo_n_init:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
    
    # Optimization
    for l in range(bo_n_init, bo_n_iters):
        if len(ix_candidates)==0:
            ix_evaled.append(-1)
            ybest_list.append(preds.ybest)
            continue

        i = preds.next(m.X[ix_candidates])

        ix = ix_candidates[i]
        preds.add(m.X[ix], ytest[ix])
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
        ybest_list.append(preds.ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' \
                                    % (l, ytest[ix], ix, preds.ybest))

    return np.asarray(ybest_list), ix_evaled

'''Random Search'''
def random_search(bo_n_iters, ytest, speed=1, do_print=False, pipeline_ixs=None, ndcgk=[None],
              is_narrow=0, narrow_list=None):
    """
    speed denotes how many random queries are performed per iteration.
    """
    
    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest_list = []
    # ndcg_list = []

    ybest = np.nan
    n_init = 0

    for l in range(bo_n_iters):
        for ll in range(speed):
            if len(ix_candidates)==0:
                ix_evaled.append(-1)
                continue

            random_rank = np.random.permutation(len(ix_candidates))

            if is_narrow and n_init<5:
                random_rank = [i for i in random_rank if random_rank[i] in narrow_list]
                ix = ix_candidates[random_rank[0]]
                if not np.isnan(ybest):
                    if ytest[ix] > ybest:
                        ybest = ytest[ix]
                    ix_evaled.append(ix)
                    ix_candidates.remove(ix)
                    n_init+=1
            else:
                ix = ix_candidates[random_rank[0]]
                
                if np.isnan(ybest):
                    ybest = ytest[ix]
                else:
                    if ytest[ix] > ybest:
                        ybest = ytest[ix]

                ix_evaled.append(ix)
                ix_candidates.remove(ix)

        ybest_list.append(ybest)

        if do_print:
            print('Iter: %d, %g [%d], Best: %g' % (l, ytest[ix], ix, ybest))

    return np.asarray(ybest_list), ix_evaled




def bo_search_with_autogluon(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, 
                              X_test, y_test, warm_start='l1', warm_starter=None,
                              pipeline_ixs=None, is_narrow=0, narrow_list=None):
    """
    Modified BO search that evaluates pipelines using AutoGluon on actual test data
    """
    import bo
    
    # Get warm start recommendations
    ix_init = warm_starter.recommend(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    # Evaluate recommended pipelines dynamically
    print(f"\n  Evaluating recommended pipelines with AutoGluon...")
    ybest_list = []
    ix_evaled = []
    
    for ix in ix_init[:bo_n_iters]:
        config = PIPELINE_CONFIGS[ix]
        acc = evaluate_single_pipeline_autogluon(X_test, y_test, config)
        
        if not np.isnan(acc):
            ix_evaled.append(ix)
            if len(ybest_list) == 0:
                ybest_list.append(acc)
            else:
                ybest_list.append(max(acc, ybest_list[-1]))
        
        if len(ybest_list) >= bo_n_iters:
            break
    
    # Pad if needed
    while len(ybest_list) < bo_n_iters:
        ybest_list.append(ybest_list[-1] if ybest_list else 0)
        ix_evaled.append(ix_evaled[-1] if ix_evaled else -1)
    
    return np.array(ybest_list), ix_evaled


def random_search_with_autogluon(bo_n_iters, X_test, y_test, speed=1, 
                                  pipeline_ixs=None, is_narrow=0, narrow_list=None):
    """
    Random search with AutoGluon evaluation
    """
    ix_candidates = list(range(len(PIPELINE_CONFIGS)))
    if is_narrow:
        ix_candidates = [i for i in ix_candidates if i in narrow_list]
    
    ybest_list = []
    ix_evaled = []
    ybest = np.nan
    
    for l in range(bo_n_iters):
        if len(ix_candidates) == 0:
            ybest_list.append(ybest if not np.isnan(ybest) else 0)
            ix_evaled.append(-1)
            continue
        
        # Random selection
        random_idx = np.random.choice(len(ix_candidates))
        ix = ix_candidates[random_idx]
        
        config = PIPELINE_CONFIGS[ix]
        acc = evaluate_single_pipeline_autogluon(X_test, y_test, config)
        
        if not np.isnan(acc):
            if np.isnan(ybest):
                ybest = acc
            else:
                ybest = max(ybest, acc)
        
        ybest_list.append(ybest if not np.isnan(ybest) else 0)
        ix_evaled.append(ix)
        ix_candidates.remove(ix)
    
    return np.array(ybest_list), ix_evaled


'''Ours Search'''
def ours_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None):

    ix_evaled = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    ybest = 0
    ybest_list = []

    # warm start
    ix_init, init_score = warm_starter.recommend_with_score(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
            
            if ytest[ix]>ybest:
                ybest = ytest[ix]
            ybest_list.append(ybest)
            
            n_init +=1
            if n_init==bo_n_iters:
                break

    while len(ybest_list)<bo_n_iters:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
    
    return np.asarray(ybest_list), ix_evaled

'''
SMAC Search
'''
try:  # pragma: no cover - optional dependency
    from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ConfigurationSpace = None  # type: ignore
    CategoricalHyperparameter = None  # type: ignore
    OrdinalHyperparameter = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from smac.scenario.scenario import Scenario  # type: ignore
    from smac.facade.smac_hpo_facade import SMAC4HPO  # type: ignore
    _SMAC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Scenario = None  # type: ignore
    SMAC4HPO = None  # type: ignore
    _SMAC_AVAILABLE = False

def smac_search(m, bo_n_init, bo_n_iters, Ytrain, Ftrain, ftest, ytest,
        do_print=False, warm_start='l1', warm_starter=None, pipeline_ixs=None,
        is_narrow=0, narrow_list=None, pipeline_names=None):
    if not _SMAC_AVAILABLE or ConfigurationSpace is None or OrdinalHyperparameter is None:
        raise ImportError(
            "SMAC is not available in this environment. Install a compatible release (for example, 'pip install smac<1.0') "
            "or disable SMAC-based search."
        )
    ix_evaled = []
    init_names = []
    ix_candidates = np.where(np.invert(np.isnan(ytest)))[0].tolist()
    all_candidates = ix_candidates.copy()
    id_evaled = []
    id_candidates = [str(i) for i in list(range(len(ix_candidates)))]
    candidate_names = [pipeline_names[i] for i in ix_candidates]
    ybest = 0
    ybest_list = []

    # warm start
    ix_init = warm_starter.recommend(ftest)
    n_init = 0
    if is_narrow:
        ix_init = [i for i in ix_init if i in narrow_list]
    
    for ix in ix_init:
        if not np.isnan(ytest[ix]):
            ix_evaled.append(ix)
            init_names.append(pipeline_names[ix])
            ix_candidates.remove(ix)
            candidate_names.remove(pipeline_names[ix])
            
            #id = str(all_candidates.index(ix))
            #id_evaled.append(id)
            #id_candidates.remove(id)
            
            if ytest[ix]>ybest:
                ybest = ytest[ix]
            ybest_list.append(ybest)
            
            n_init +=1
            if n_init==bo_n_init:
                break
    
    # optimization
    init_cs = ConfigurationSpace()
    #init_ids = OrdinalHyperparameter('name', [str(i) for i in id_evaled], default_value=str(id_evaled[0]))
    init_ids = OrdinalHyperparameter('name', init_names, default_value=init_names[0])
    init_cs.add_hyperparameter(init_ids)

    optim_cs = ConfigurationSpace()
    #optim_ids = OrdinalHyperparameter('name', [str(i) for i in id_candidates], default_value=str(id_candidates[0]))
    optim_ids = OrdinalHyperparameter('name', candidate_names, default_value=candidate_names[0])
    optim_cs.add_hyperparameter(optim_ids)
    scenario = Scenario({'run_obj':'quality', 'runcount-limit':bo_n_iters, 'cs':optim_cs, 'deterministic':'true',
        'initial_incumbent':"DEFAULT"})

    def cfg_ours(cfg):
        #print(cfg['name'])
        ix = pipeline_names.index(cfg['name'])
        #id = int(cfg['names'])
        #ix = all_candidates[int(cfg['name'])]
        
        ix_evaled.append(ix)
        if ytest[ix]>ybest_list[-1]:
            ybest_list.append(ytest[ix])
        else:
            ybest_list.append(ybest_list[-1])

        return np.float32(ytest[ix])

    init_confs = []
    for name in init_names:
        conf = init_cs.sample_configuration()
        conf['name'] = name
        init_confs.append(conf)
    

    smac = SMAC4HPO(scenario=scenario, rng=0, tae_runner=cfg_ours, 
        smbo_kwargs={'min_samples_model':10},
        initial_design_kwargs={'configs':init_confs, 'init_budget':bo_n_init}
        )#initial_design_kwargs={"cs": init_cs, 'init_budget':100, 'ta_run_limit':100},)
    
    incumbent = smac.optimize()
    
    while len(ybest_list)<bo_n_iters:
        ix_evaled.append(ix_evaled[-1])
        ybest_list.append(ybest_list[-1])
        
    return np.asarray(ybest_list[bo_n_init:]), ix_evaled[bo_n_init:]