# data handling
import pandas as pd
import numpy as np
from StandardConfig import timingmethod
from StandardConfig import find_folderpath
from StandardConfig import make_directory
# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
# visualization
from sklearn.tree import export_graphviz
import graphviz
from Translate_TMDrefined import aa_numeric_by_scale
from datetime import date
import matplotlib.pyplot as plt


class ForestTMDrefind:

    def __init__(self, df_train_instance_parameters, df_train_labels, best_params, param_dist, search_cv, model,
                 job_name, path_forest, start_tmd):
        self.df_train_instance_parameters = df_train_instance_parameters
        self.df_train_labels = df_train_labels
        self.best_params = best_params
        self.param_dist = param_dist
        self.search_cv = search_cv
        self.model = model
        self.job_name = job_name
        self.path_forest = path_forest
        self.start_tmd = start_tmd

    # random forest algorithm --> search and train with best params
    # __________________________________________________________________________________________________________________
    @classmethod
    @timingmethod
    def make_forest(cls, df_train_instance_parameters, df_train_labels, job_name, start_tmd=True,
                    n_jobs=1, param_grid=None):
        # initialize the RandomForestClassifier as clf
        clf = RandomForestClassifier()

        # parameter distribution
        if param_grid is None:
            param_grid = {'n_estimators': np.linspace(100, 500, 51, dtype=int),
                          'max_depth': np.linspace(3, 30, 28, dtype=int),
                          'max_leaf_nodes': np.linspace(10, 20, 11, dtype=int),
                          'class_weight': ["balanced"],
                          'bootstrap': [True],
                          'n_jobs': [n_jobs],
                          'criterion': ["gini"]
                          }
            # 51*28*11 = 14,586 different combinations of current parameters, would be too time intensive!

        search_cv = (HalvingGridSearchCV(clf, param_grid, resource='n_samples', max_resources="auto",
                                         scoring='neg_mean_absolute_error').fit(df_train_instance_parameters.
                                                                                to_numpy().tolist(), df_train_labels.
                                                                                to_numpy().tolist()))
        path, sep = find_folderpath()
        date_today = date.today()
        path_forest = f"output_forest_model_[{job_name}]_{date_today}"
        make_directory(path_forest)

        # output of search:
        best_params = search_cv.best_params_
        with open(f"{path_forest}{sep}{job_name}_best_params.txt", "w") as file:
            file.write(f"{best_params}")

        # new model with best_params
        clf_best = RandomForestClassifier(**best_params)
        clf_best.fit(df_train_instance_parameters.to_numpy().tolist(), df_train_labels.to_numpy().tolist())

        # print best parameters
        print(f"given parameters: {param_grid} with HalvingGridSearchCV gave following best-performing parameter combi:"
              f"best params: {best_params}")

        return cls(df_train_instance_parameters, df_train_labels, best_params, param_grid, search_cv, clf_best,
                   job_name, path_forest, start_tmd)

    # analysis functions
    # __________________________________________________________________________________________________________________
    def fetch_a_tree(self):
        path, sep = find_folderpath()
        make_directory("output_tree")
        tree_number = np.random.randint(self.best_params["n_estimators"], size=1)[0]
        tree = self.model.estimators_[tree_number]
        dot_data = export_graphviz(tree,
                                   feature_names=self.df_train_instance_parameters.columns,
                                   out_file=None,
                                   impurity=True,
                                   proportion=True,
                                   rounded=True,
                                   filled=True)

        graph = graphviz.Source(dot_data)
        graph.format = "png"
        date_today = date.today()
        graph.render(f"{path}{sep}output_tree{sep}{self.job_name}_sample_tree_dot_{date_today}", view=True)

    def hyperparameter_summary(self, save_table=False):
        columns = [f"param_{name}" for name in self.param_dist.keys()]
        columns += ["mean_test_error", "std_test_error"]
        cv_results = pd.DataFrame(self.search_cv.cv_results_)
        cv_results["mean_test_error"] = -cv_results["mean_test_score"]
        cv_results["std_test_error"] = cv_results["std_test_score"]
        cv_results_final = cv_results[columns].sort_values(by="mean_test_error")
        if save_table:
            date_today = date.today()
            cv_results_final.to_excel(f"{self.path_forest}{sep}{self.job_name}_hyperparameter_{date_today}.xlsx")
        return cv_results_final

    def feature_importance(self):
        # Create a series containing feature importance from the model and feature names from the training data

        # setup for colored bar plot and legend referring to the meaning based on AAOntology categories [Breimann, 24c]
        aaontology_colors_hex = {'ASA/Volume': "#3680b4",
                                 'Polarity': "#fddc22",
                                 'Structure-Activity': "#8c5e56",
                                 'Composition': "#ff9232",
                                 'Others': "#7f7f7f",
                                 'Shape': "#37c0ce",
                                 'Conformation': "#3d9f47",
                                 'Energy': "#d54141"
                                 }

        path_to_scales_df = f"{path.split("scripts")[0]}_scales{sep}scales_cat.xlsx"
        # scale_cat allows for mapping of categories
        scale_cat_df = pd.read_excel(path_to_scales_df).set_index("scale_id")

        feature_importance = pd.Series(self.model.feature_importances_,
                                       index=self.df_train_instance_parameters.columns).sort_values(ascending=False)
        data = feature_importance.tolist()
        scale_labels = feature_importance.index.tolist()   # scale_labels are the scale_id in scale_cat
        # generate mapping colors on bars for each scale

        list_aao_color_for_bar = []
        for labels in scale_labels:
            color_tag = scale_cat_df.loc[labels, "category"]
            get_aao_color = aaontology_colors_hex[color_tag]
            list_aao_color_for_bar.append(get_aao_color)

        # Plot a simple bar chart
        fig, ax = plt.subplots()
        ax.bar(scale_labels, data, color=list_aao_color_for_bar)
        ax.set_xticklabels(scale_labels, rotation=45, ha="right", fontweight="bold")
        ax.set_title("Feature Importance", fontsize=20, fontweight="bold")
        ax.spines[['right', 'top']].set_visible(False)
        # for legend
        labels_legend = list(aaontology_colors_hex.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=aaontology_colors_hex[label]) for label in labels_legend]
        ax.legend(handles, labels_legend)
        # saving
        date_today = date.today()
        plt.savefig(f"{self.path_forest}{sep}{self.job_name}_feature_importance_{date_today}.png", dpi=400,
                    bbox_inches="tight")

    def predict_labels(self, df_pred_instance_parameters):
        pred_labels = self.model.predict(df_pred_instance_parameters.to_numpy().tolist())
        return pred_labels

    def test_predict_quality(self, label_test, label_pred, cm_save=False):

        # general info about performance
        accuracy = accuracy_score(label_test, label_pred)
        precision = precision_score(label_test, label_pred)
        recall = recall_score(label_test, label_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        # Create the confusion matrix
        cm = confusion_matrix(label_test, label_pred)
        if cm_save:
            date_today = date.today()
            cm.figure_.savefig(f"{self.path_forest}{sep}{self.job_name}_confusion_matrix_{date_today}.png")
        else:
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    # predict based on sequences + initial predicted JMD|TMD intersection
    # __________________________________________________________________________________________________________________
    def pred_from_seq(self, sequence, pos_intersect):
        pass


# debugging
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    path, sep = find_folderpath()
    list_scale_names = ["BURA740101", "CHAM830104"]
    label_df_test = (pd.read_excel(f"{path.split("scripts")[0]}example{sep}am_label_single_stop_pos_TMD_thresh=3.xlsx")
                     .set_index("ID"))

    scale_df, id_tags_df = aa_numeric_by_scale(feature_df=label_df_test[["window_left", "window_right"]],
                                               label_df=label_df_test["label"], scale_df_filter=list_scale_names)

    param_dist_testi = {'n_estimators': np.linspace(100, 500, 3, dtype=int),
                        "max_depth": [3, None],
                        "min_samples_split": [5, 10],
                        "n_jobs": [4],
                        'criterion': ["gini"]
                        }
    example_forest = ForestTMDrefind.make_forest(scale_df, id_tags_df, param_grid=param_dist_testi, job_name="test")
    example_forest.fetch_a_tree()
    print(example_forest.hyperparameter_summary())
    example_forest.feature_importance()
