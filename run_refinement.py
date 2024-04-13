# general utility imports
import glob
import pandas as pd
# ml imports
import scripts.ML_TMDrefined as ml_ref
import scripts.StandardConfig as stdc
from scripts.Translate_TMDrefined import aa_numeric_by_scale


def run(df: pd.DataFrame, start_pos_col: str, stop_pos_col: str, seq_col: str,  job_name: str,
        lower_len_tmd_border: int = 18, upper_len_tmd_border: int = 30):
    # check input
    # __________________________________________________________________________________________________________________
    if not isinstance(df, pd.DataFrame):
        raise TypeError("needs to be pd.DataFrame")
    for cols in [start_pos_col, stop_pos_col, seq_col]:
        if not isinstance(cols, str):
            raise TypeError(f"{cols} must be a string!")
    for cols in [start_pos_col, stop_pos_col, seq_col]:
        if cols not in df.columns:
            raise ValueError(f"{cols} not a df column!")
    for number_col in [start_pos_col, stop_pos_col]:
        if str(df[number_col].dtypes).find("float") != 0:
            if str(df[number_col].dtypes).find("int") != 0:
                raise TypeError(f"{number_col} must be float or int")
    if str(df[seq_col].dtypes).find("object") != 0:
        raise TypeError(f"{seq_col} must be object")

    # general pathing
    # __________________________________________________________________________________________________________________
    path, path_module, sep = stdc.find_folderpath()
    stdc.make_directory(job_name)

    # get training labels
    # __________________________________________________________________________________________________________________
    path_labels = f"{path}{sep}_train_3_models_data{sep}"
    n_labels_paths = glob.glob(f"{path_labels}mean_weight_balance{sep}mean_N_balance{sep}*.xlsx")  # test, then train
    print(n_labels_paths)
    c_labels_paths = glob.glob(f"{path_labels}mean_weight_balance{sep}mean_C_balance{sep}*.xlsx")  # test, then train
    scales_path = f"{path_labels}KMeans_scales_norm.xlsx"

    # get all DataFrame
    # __________________________________________________________________________________________________________________
    test_train_n_list_df = []
    test_train_c_list_df = []
    df_scales = pd.read_excel(scales_path).set_index("AA").columns.tolist()  # features
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

    if "weightsx" in test_train_n_list_df[0].columns.tolist():
        weight_n_df = test_train_n_list_df[0]["weights"]
    else:
        weight_n_df = None

    n_forest = ml_ref.ForestTMDrefind.make_forest(test_train_n_list_df[0][['window_left', 'window_right']],
                                                  test_train_n_list_df[0]["label"], df_train_weights=weight_n_df,
                                                  scales_list=df_scales, job_name=f"{job_name}_n_forest", n_jobs=-1,
                                                  param_grid=param_grid_n, model_retrains=30)
    n_forest.hyperparameter_summary(save_table=True)
    n_forest.feature_importance()
    n_test_labels_pred = n_forest.predict_labels(test_train_n_list_df[1][['window_left', 'window_right']])
    n_forest.test_predict_quality(label_test=test_train_n_list_df[1]["label"], label_pred=n_test_labels_pred[0],
                                  cm_save=True)

    param_grid_c = {'bootstrap': [True],
                    'class_weight': ['balanced_subsample'],
                    'criterion': ['entropy'],
                    'max_depth': [26],
                    'max_leaf_nodes': [260],
                    'max_samples': [0.3],
                    'n_estimators': [640],
                    'n_jobs': [-1]}

    if "weightsx" in test_train_c_list_df[0].columns.tolist():
        weight_c_df = test_train_c_list_df[0]["weights"]
    else:
        weight_c_df = None

    c_forest = ml_ref.ForestTMDrefind.make_forest(test_train_c_list_df[0][['window_left', 'window_right']],
                                                  test_train_c_list_df[0]["label"], df_train_weights=weight_c_df,
                                                  scales_list=df_scales, job_name=f"{job_name}_c_forest", n_jobs=-1,
                                                  param_grid=param_grid_c, model_retrains=30, start_tmd=False)
    c_forest.hyperparameter_summary(save_table=True)
    c_forest.feature_importance()
    c_test_labels_pred = c_forest.predict_labels(test_train_c_list_df[1][['window_left', 'window_right']])
    c_forest.test_predict_quality(label_test=test_train_c_list_df[1]["label"], label_pred=c_test_labels_pred[0],
                                  cm_save=True)

    # generate new DataFrame
    # __________________________________________________________________________________________________________________
    df_filter = df[[start_pos_col, stop_pos_col, seq_col]]
    df_filter = df_filter.dropna(how="any")

    list_tmd_refined = []
    columns_tmd_refined = ["index", "len_tmd", start_pos_col, stop_pos_col, "jmd_n", "tmd", "jmd_c", seq_col]
    for index, rows in df_filter.iterrows():
        set_keep = True
        pred_n = n_forest.pred_from_seq(index, rows[2], int(rows[0]), show_plot=False)  # default is False
        pred_c = c_forest.pred_from_seq(index, rows[2], int(rows[1]), show_plot=False)
        start_pos = pred_n[1]
        stop_pos = pred_c[1]
        len_tmd = stop_pos-start_pos
        # check length criterium of the TMD
        if (len_tmd < lower_len_tmd_border) or (len_tmd > upper_len_tmd_border):
            list_all_proba_combos = []
            for index_n, rows_n in pred_n[2].iterrows():
                for index_c, rows_c in pred_c[2].iterrows():
                    len_tmd_check = abs(rows_n[0]-rows_c[0])
                    proba_sum = rows_n[1]+rows_c[1]
                    list_all_proba_combos.append([len_tmd_check, proba_sum, rows_n[0], rows_c[0]])
            df_proba_len_tmd = (pd.DataFrame(list_all_proba_combos, columns=["len_tmd", "sum_proba", "start", "stop"]).
                                sort_values("sum_proba", ascending=False))
            pos_df_proba_len_tmd = 0
            while pos_df_proba_len_tmd < len(df_proba_len_tmd):
                if df_proba_len_tmd.iloc[pos_df_proba_len_tmd, 0] >= lower_len_tmd_border:
                    if df_proba_len_tmd.iloc[pos_df_proba_len_tmd, 0] <= upper_len_tmd_border:
                        start_pos = int(df_proba_len_tmd.iloc[pos_df_proba_len_tmd, 2])
                        stop_pos = int(df_proba_len_tmd.iloc[pos_df_proba_len_tmd, 3])
                        len_tmd = stop_pos-start_pos
                        break
                pos_df_proba_len_tmd += 1
            if pos_df_proba_len_tmd == len(df_proba_len_tmd):
                set_keep = False

        jmd_n = rows[2][start_pos-11: start_pos-1]
        if start_pos - 11 < 0:
            jmd_n = rows[2][0: start_pos - 1]
        tmd = rows[2][start_pos-1: stop_pos]
        jmd_c = rows[2][stop_pos: stop_pos+10]
        if set_keep:
            list_tmd_refined.append([index, len_tmd, start_pos, stop_pos, jmd_n, tmd, jmd_c, rows[2]])
    df_tmd_refined = pd.DataFrame(list_tmd_refined, columns=columns_tmd_refined).set_index("index")
    df_tmd_refined.to_excel(f"{path}{sep}{job_name}{sep}{job_name}_tmd_refined.xlsx")


if __name__ == "__main__":
    # "/home/freiherr/PycharmProjects/TMDrefined/_training_data/arithmetic_mean_all_annots_for_refining.xlsx"
    df_input = pd.read_excel("/home/freiherr/PycharmProjects/TMDrefined/_train_3_models_data/mean_weight/arithmetic_mean_all_annots_for_refining.xlsx").set_index("entry").head(10)
    run(df=df_input,
        start_pos_col="start_pos_TMD",
        stop_pos_col="stop_pos_TMD",
        seq_col="sequence",
        job_name="mean_test_weights_balance")
