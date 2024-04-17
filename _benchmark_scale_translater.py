import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scripts.ML_TMDrefined as ml_ref
import scripts.StandardConfig as stdc
import seaborn as sns


def run_benchmark_translater(job_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame, param_grid: dict,
                             start_tmd: bool = True):
    weight_start = [1.0001, 1.001, 1.01, 1.1, 3, 5, 7, 10, 100, 1000]
    weight_step = [0, 0.01, 0.1, 1, 3, 5, 7, 10, 100, 1000]
    path, path_module, sep = stdc.find_folderpath()
    path_labels = f"{path}{sep}_train_3_models_data{sep}"
    scales_path = f"{path_labels}KMeans_scales_norm.xlsx"
    df_scales = pd.read_excel(scales_path).set_index("AA").columns.tolist()

    list_f1 = []
    for starts in weight_start:
        row_step = []
        for steps in weight_step:
            forest = ml_ref.ForestTMDrefind.make_forest(df_train[['window_left', 'window_right']],
                                                        df_train["label"], df_train_weights=df_train["weights"],
                                                        scales_list=df_scales, job_name=f"{job_name}_n_forest",
                                                        n_jobs=-1, param_grid=param_grid, model_retrains=1,
                                                        start_tmd=start_tmd, weight_start=starts, weight_step=steps)
            test_labels_pred = forest.predict_labels(df_test[['window_left', 'window_right']])[0]
            f1 = forest.test_predict_benchmark(label_test=df_test["label"], label_pred=test_labels_pred)[3]
            row_step.append(f1)
        list_f1.append(row_step)
    print(list_f1)
    data_df = pd.DataFrame(list_f1)

    # heatmap generation
    # plotting a triangle correlation heatmap
    cmap = sns.color_palette('magma', as_cmap=True)
    dataplot = sns.heatmap(data_df, annot=True, center=0.75, vmin=0.5, vmax=1, cmap=cmap, linewidths=3,
                           linecolor='white', fmt=".2f")
    plt.title(f"{job_name}\n", fontsize=14, fontweight="bold")
    dataplot.set_xticklabels(weight_step)
    plt.yticks(rotation=0)
    dataplot.set_yticklabels(weight_start)
    # displaying heatmap
    plt.savefig(f"{job_name}.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    param_grid_n = {'bootstrap': [True],
                    'class_weight': ['balanced_subsample'],
                    'criterion': ['entropy'],
                    'max_depth': [40],
                    'max_leaf_nodes': [250],
                    'max_samples': [0.3],
                    'n_estimators': [600],
                    'n_jobs': [-1]}

    job_name = "Scale translation: weight-factors influence on performance (f1 score)"
    df_train_test = pd.read_excel("/home/freiherr/PycharmProjects/TMDrefined/_train_3_models_data/mean_weight_easy8/mean_N/mean_test_df_N_balance8.xlsx")
    df_test_test = pd.read_excel("/home/freiherr/PycharmProjects/TMDrefined/_train_3_models_data/mean_weight_easy8/mean_N/mean_val_df_N_balance8.xlsx")

    run_benchmark_translater(job_name, df_train_test, df_test_test, param_grid_n)

