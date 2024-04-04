# imports
import pandas as pd
import numpy as np
import scripts.ML_TMDrefined as ml_ref
import scripts.Translate_TMDrefined as t_ref
import scripts.StandardConfig as stdc
import glob


def run(df: pd.DataFrame, job_name: str):
    # general pathing
    # __________________________________________________________________________________________________________________
    path, path_module, sep = stdc.find_folderpath()

    # get training labels
    # __________________________________________________________________________________________________________________
    path_labels = f"{path}{sep}N_out_refinement{sep}"
    n_labels_paths = glob.glob(f"{path_labels}test_train_N{sep}*.xlsx")  # test, then train
    c_labels_paths = glob.glob(f"{path_labels}test_train_C{sep}*.xlsx")  # test, then train
    top60_brei_path = f"{path_labels}CPP_top60_n=1{sep}norm_top_of_60.xlsx"

    # get all DataFrame
    # __________________________________________________________________________________________________________________
    test_train_n_list_df = []
    test_train_c_list_df = []
    top60_1_features = pd.read_excel(top60_brei_path).set_index("AA").columns.tolist()  # features
    for paths, lists in zip([n_labels_paths, c_labels_paths], [test_train_n_list_df, test_train_c_list_df]):
        for sub_paths in paths:
            df_labels = pd.read_excel(sub_paths).set_index("ID")
            lists.append(df_labels)

    # fit the models
    # __________________________________________________________________________________________________________________
    param_grid_n = {'bootstrap': [True],
                    'class_weight': ['balanced_subsample'],
                    'criterion': ['entropy'],
                    'max_depth': [40],
                    'max_leaf_nodes': [250],
                    'max_samples': [0.3],
                    'n_estimators': [600],
                    'n_jobs': [-1]}
    n_forest = ml_ref.ForestTMDrefind.make_forest(test_train_n_list_df[1][['window_left', 'window_right']],
                                                  test_train_n_list_df[1]["label"], scales_list=top60_1_features,
                                                  job_name=f"job_name{sep}n_forest", n_jobs=-1, param_grid=param_grid_n,
                                                  model_retrains=30)
    n_forest.hyperparameter_summary(save_table=True)
    n_forest.feature_importance()
    n_test_labels_pred = n_forest.predict_labels(test_train_n_list_df[0][['window_left', 'window_right']])
    n_forest.test_predict_quality(label_test=test_train_n_list_df[0]["label"], label_pred=n_test_labels_pred[0],
                                  cm_save=True)

    param_grid_c = {'bootstrap': [True],
                    'class_weight': ['balanced_subsample'],
                    'criterion': ['entropy'],
                    'max_depth': [26],
                    'max_leaf_nodes': [260],
                    'max_samples': [0.3],
                    'n_estimators': [640],
                    'n_jobs': [-1]}
    c_forest = ml_ref.ForestTMDrefind.make_forest(test_train_c_list_df[1][['window_left', 'window_right']],
                                                  test_train_c_list_df[1]["label"], scales_list=top60_1_features,
                                                  job_name=f"job_name{sep}c_forest", n_jobs=-1, param_grid=param_grid_c,
                                                  model_retrains=30, start_tmd=False)
    c_forest.hyperparameter_summary(save_table=True)
    c_forest.feature_importance()
    c_test_labels_pred = c_forest.predict_labels(test_train_c_list_df[0][['window_left', 'window_right']])
    c_forest.test_predict_quality(label_test=test_train_c_list_df[0]["label"], label_pred=c_test_labels_pred[0],
                                  cm_save=True)



if __name__ == "__main__":
    df = ""
    run(df, "tmd_refined_N_out")