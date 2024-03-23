import pandas as pd
import numpy as np
from .StandardConfig import timingmethod
from .StandardConfig import find_folderpath
import math
from pandas.api.types import is_string_dtype

# localization of script
# ______________________________________________________________________________________________________________________
path, sep = find_folderpath()


# creating the input for the model
# ______________________________________________________________________________________________________________________

@timingmethod
def aa_numeric_by_scale(feature_df: pd.DataFrame, label_df: pd.DataFrame, scale_df_filter: list = None,
                        mode: str = "weighted"):
    # check function block
    # ______________________________________________________________________________

    # modes of agglomerating label window sides
    modes_list = ["sum", "norm", "weighted"]
    if str(mode) not in modes_list:
        mode = "weighted"

    # check correct label_df
    if not isinstance(feature_df, pd.DataFrame):
        raise ValueError("needs to be pd.DataFrame!")
    elif len(feature_df.columns.tolist()) != 2:
        raise ValueError("pd.DataFrame needs to be two columns!")
    feature_df = feature_df.dropna()
    for col in feature_df.columns.tolist():
        if is_string_dtype(feature_df[col]) is False:
            raise ValueError("only str value allowed = sequence tags")

    # check intersect between label and feature df
    check_index_feature = feature_df.index.tolist()
    check_index_label = label_df.index.tolist()
    shared_index = [value for value in check_index_feature if value in check_index_label]
    if len(shared_index) == 0:
        raise ValueError("Make sure feature_df and label_df have the same index, no shared indices!")
    else:
        feature_df = feature_df.loc[shared_index]
        label_df = label_df.loc[shared_index]

    # normalized scales
    path_to_scales_df = f"{path.split("scripts")[0]}_scales{sep}scales.xlsx"
    scale_df = pd.read_excel(path_to_scales_df).set_index("AA")

    # filter scale_df
    scales_df_contains = scale_df.columns.tolist()
    if not isinstance(scale_df_filter, list):
        scale_df_filter = scale_df
    accepted_scales = []
    for tags in scale_df_filter:
        if str(tags) in scales_df_contains:
            accepted_scales.append(str(tags))
    if len(accepted_scales) == 0:
        scale_df_filter = scale_df
    else:
        scale_df_filter = scale_df[accepted_scales]

    # aa translater to scales
    # _______________________________________________________________________________
    feature_list = feature_df.to_numpy().tolist()
    feature_index_list = feature_df.index.tolist()
    list_diff_by_scales = []

    for index_label, parts in zip(feature_index_list, feature_list):
        order_aa = scale_df_filter.T.columns.tolist()
        list_intermediate_scale_translate = []
        for index, row in scale_df_filter.T.iterrows():
            parts_list = []
            # for weighted the algorithm has to be mirrored for each mirror side, so this is essential!
            flag_reverse = 1
            for part in parts:  # if part is nan --> can be interpreted as float
                # part_list has float values!
                part_list = [row.to_numpy().tolist()[order_aa.index(letters)] for letters in part]
                # check which mode was selected
                if mode == "sum":
                    value = sum(part_list)
                elif mode == "norm":
                    value = sum(part_list)/len(part_list)
                else:
                    if flag_reverse == 1:
                        part_list.reverse()
                        flag_reverse = 0  # turn flag off
                    list_ln_linspace = np.linspace(3, 2+len(part), num=len(part))
                    list_ln = [math.log(num) for num in list_ln_linspace]
                    value_pre = (sum([(letters/ln) for letters, ln in zip(part_list, list_ln)]))
                    if value_pre == 0:
                        value = 0       # avoid ZeroDivision Error
                    else:
                        value = value_pre/sum(list_ln)
                parts_list.append(value)
            list_intermediate_scale_translate.append(parts_list[0]-parts_list[1])
        list_intermediate_scale_translate.append(index_label)
        list_diff_by_scales.append(list_intermediate_scale_translate)
    columns_df_diff_by_scales = scale_df_filter.columns.tolist()
    columns_df_diff_by_scales.append("ID")
    df_diff_by_scales = pd.DataFrame(list_diff_by_scales, columns=columns_df_diff_by_scales).set_index("ID")
    return df_diff_by_scales, label_df


# probably redundant at this point, but might come in handy for back up dataset for testing!
def balanced_test_set_generator(label_df: pd.DataFrame, target_column: str, fold_split: int = 5,
                                split_test_train: float = 0.2, set_random_state=None):
    len_label_df_pre_filter = label_df.shape[0]
    label_df = label_df.dropna(how="any", axis=0)
    # split label_df
    len_label_df = label_df.shape[0]

    print(f"""
    Input data beginning: {len_label_df_pre_filter} rows
    Post filtering:       {len_label_df} rows
    """)

    len_segment = round(len_label_df/fold_split, 0)

    count_test_rows_per_segment = round(len_segment*split_test_train, 0)
    if int(count_test_rows_per_segment) == 0:
        raise ValueError("increase the split_test_train variable or decrease the balance_split variable, "
                         "currently 0 instances are used for test set")

    label_df = label_df.sort_values(by=target_column, ascending=True)
    slice_list = []
    i=0
    while i < len_label_df:
        slice_df = label_df.iloc[int(i):int(i+len_segment), :]
        slice_samples = slice_df.sample(n=int(count_test_rows_per_segment), random_state=set_random_state)
        slice_list.append(slice_samples)
        i += len_segment+1

    df_test = pd.concat(slice_list)
    df_train = label_df.drop(df_test.index.tolist(), axis=0)
    return df_train, df_test


# debugging
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    # input
    list_scale_names = ["BURA740101", "CHAM830104"]
    label_df_test = (pd.read_excel(f"{path.split("scripts")[0]}example{sep}am_label_single_stop_pos_TMD_thresh=3.xlsx")
                     .set_index("ID"))

    scale_df, id_tags_df = aa_numeric_by_scale(feature_df=label_df_test[["window_left", "window_right"]],
                                               label_df=label_df_test["label"], scale_df_filter=list_scale_names)
    print(scale_df)
    print(id_tags_df)

    df_train_test, df_test_test = balanced_test_set_generator(label_df_test, "norm_intersect_pos", fold_split=101)
    print(df_train_test.describe())
    print(df_test_test.describe())
