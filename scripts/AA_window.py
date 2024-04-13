import pandas as pd
import numpy as np


# __main__ methods
# ______________________________________________________________________________________________________________________
def get_aa_window(window_size: int, aa_seq: str, aa_position: int, start_pos: bool):
    """
    generates window-slices of the sequence

    Parameters
    __________
    window_size : sets the slice of the window on each side of the position: e.g.: window size 4 = LLLL | KKKK
    aa_seq : amino acid sequence
    aa_position : position where the cut occurs

    Returns
    _______
    window of set window_size left and right of the slicing position
    """

    start = aa_position - window_size - 1
    start_of_stop = aa_position - 1
    if start_pos is False:
        start = start + 1
        start_of_stop = start_of_stop + 1

    list_left_right_window = []
    for pos in [start, start_of_stop]:
        if pos <= -4:                              # no AA at label window side
            list_left_right_window.append(np.nan)
        elif pos < 0:                              # less than 4 AA at label window side
            list_left_right_window.append(aa_seq[0: pos + window_size])
        else:                                      # 4 AA at label window side
            list_left_right_window.append(aa_seq[pos: pos + window_size])

    if list_left_right_window is []:
        list_left_right_window = [np.nan, np.nan]

    return list_left_right_window


def get_aa_window_labels(window_size: int, aa_seq: str, name_label: str, tmd_jmd_intersect: int, start_pos: bool,
                         column_pos_in_seq: str = "pos_in_seq", more_columns: dict = None, range_window: int = None,
                         start_set_positive: bool = False):
    """
    Generates positive and negative labels

    Parameters
    __________
    window size : sets the slice of the window on each side of the position: e.g.: window size 4 = LLLL | KKKK
    aa_seq : amino acid sequence
    name_label : e.g. protein-name
    tmd_jmd_intersect : start or stop position for membrane domain sequence of a protein (can be used generally)
    start_pos : is it the first amino acid of the sequence segment or the last amino acid?
    *args_columns : add more columns to dataframe

    Returns
    _______
    df_list_labels : pd.DataFrame with labels of a specific protein
    """

    columns_window = ["ID", "window_left", "window_right", "label", column_pos_in_seq, "norm_intersect_pos"]
    if more_columns is not None:
        columns_window.extend(list(more_columns.keys()))

    if not isinstance(start_set_positive, bool):
        label_set = 0
    elif not start_set_positive:
        label_set = 0
    else:
        label_set = 1

    # generate first-negative-label
    window_seq = get_aa_window(window_size, aa_seq=aa_seq, aa_position=tmd_jmd_intersect, start_pos=start_pos)
    list_labels = [[f"{name_label}__0", window_seq[0], window_seq[1], label_set, tmd_jmd_intersect,
                    tmd_jmd_intersect/len(aa_seq)]]
    if more_columns is not None:
        list_labels.extend(list(more_columns.values()))

    # generate negative N/C-term label
    if range_window is None:
        range_window = window_size
    elif not isinstance(range_window, int):
        range_window = window_size

    i = 1
    while i < int(range_window):
        left_shift_window_seq = get_aa_window(window_size, aa_seq=aa_seq, aa_position=tmd_jmd_intersect - i,
                                              start_pos=start_pos)
        right_shift_window_seq = get_aa_window(window_size, aa_seq=aa_seq, aa_position=tmd_jmd_intersect + i,
                                               start_pos=start_pos)
        sublist = [
            [f"{name_label}__-{i}", left_shift_window_seq[0], left_shift_window_seq[1], 0, tmd_jmd_intersect - i,
             (tmd_jmd_intersect-i)/len(aa_seq)],
            [f"{name_label}__{i}", right_shift_window_seq[0], right_shift_window_seq[1], 0, tmd_jmd_intersect + i,
             (tmd_jmd_intersect+i)/len(aa_seq)]]
        if more_columns is not None:
            sublist.extend(list(more_columns.values()))
        list_labels.extend(sublist)
        i += 1
    df_list_labels = pd.DataFrame(list_labels, columns=columns_window)
    return df_list_labels


def get_df_slice(id_string, df):
    df_slice_from_id = df.reset_index()[df.reset_index()["ID"].str.contains(id_string)]
    arr_slice_positives = np.array([str(id_tag).split("__")[0] for id_tag in df_slice_from_id["ID"].tolist()])
    filter_list_slice_label = np.where(arr_slice_positives == id_string)
    df_slice_from_id = df_slice_from_id.iloc[filter_list_slice_label]
    return df_slice_from_id


def label_describe(df):
    """
    purely an attribute for get_aa_window_df() and modify_label_by_ident_column() --> how many positive labels?

    Parameters
    __________
    df : label_df(_modified)

    Returns
    _______
    df_label_describe : pd.DataFrame.describe() method applied slice wise on original label_df
    """
    df_label_search_list = list(dict.fromkeys([str(index).split("__")[0] for index in df.index.tolist()]))

    list_describe = []
    for query in df_label_search_list:
        df_slice = get_df_slice(query, df)
        row, column = df_slice.shape
        count_positives = df_slice["label"].to_numpy().tolist().count(1)
        percent_positives = f"{count_positives} / {row}"
        list_describe.append([query, count_positives, percent_positives])
    label_wise_columns = ["ID", "positive_count", "positive_percent"]
    df_label_wise = pd.DataFrame(list_describe, columns=label_wise_columns)

    describe_all_labels_columns = ["average_positive", "min", "max", "ID_count"]
    labels_pos = df_label_wise["positive_count"].to_numpy().tolist()
    df_slice = get_df_slice(df_label_search_list[0], df)
    row, column = df_slice.shape
    list_describe_all_labels = [f"{round(np.mean(labels_pos), 2)} / {row}", f"{np.min(labels_pos)} / {row}",
                                f"{np.max(labels_pos)} / {row}", f"{len(df_label_search_list)}"]
    df_label_describe = pd.Series(list_describe_all_labels).set_axis(describe_all_labels_columns)
    return df_label_wise, df_label_describe
# ______________________________________________________________________________________________________________________


class AAwindowrizer:

    def __new__(cls, df):
        return super().__new__(cls)

    def __init__(self, df):
        self.df = df


    @classmethod
    def get_aa_window_df(cls, window_size: int, df, column_id: str, column_seq: str, column_aa_position: str,
                         start_pos: bool = True, column_pos_in_seq: str = None, more_columns_from_df: list = None,
                         range_window: int = None, start_set_positive: bool = False):
        """
        Parameters
        __________
        window_size = defines AA_window that is shifted --> given size is mirrored for N and C term
        df : pd.DataFrame
        column_id : column name of sequence ID e.g. UniProt entry
        column_seq : column name with the full AA_seq
        column_aa_position : column name with the TMD/JMD intersection within the AA (-1 since count from 1, not from 0)
        more_columns_from_df : add more columns for further processing

        Returns
        _______
        df with the sequence windows of positive label and negative labels
        """

        aa_window_labeled_sub_df = None

        list_id = df[column_id].to_numpy().tolist()
        list_seq = df[column_seq].to_numpy().tolist()
        list_position = df[column_aa_position].to_numpy().tolist()

        list_aa_window_labeled = []
        if column_pos_in_seq is None:
            column_pos_in_seq = column_aa_position
        for id_tag, seq, pos in zip(list_id, list_seq, list_position):
            more_columns_entry = None
            # prevent NaN values from crashing the program
            if isinstance(id_tag, (str, int)) and isinstance(seq, str) and isinstance(pos, (int, float)):
                if more_columns_from_df is not None:
                    dict_more_columns = {}
                    for key_columns_entries in more_columns_from_df:
                        dict_more_columns[key_columns_entries] = df.set_index(column_id).loc(id_tag,
                                                                                             key_columns_entries)
                    more_columns_entry = dict_more_columns
                aa_window_labeled_sub_df = get_aa_window_labels(window_size=window_size, aa_seq=seq, name_label=id_tag,
                                                                tmd_jmd_intersect=int(pos), start_pos=start_pos,
                                                                more_columns=more_columns_entry,
                                                                column_pos_in_seq=column_pos_in_seq,
                                                                range_window=range_window,
                                                                start_set_positive=start_set_positive)
                aa_window_labeled = aa_window_labeled_sub_df.to_numpy().tolist()
                list_aa_window_labeled.extend(aa_window_labeled)

        if aa_window_labeled_sub_df is not None:
            column_name = aa_window_labeled_sub_df.columns
            df_aa_window_labeled = pd.DataFrame(list_aa_window_labeled, columns=column_name).set_index(column_name[0])
            df_aa_window_labeled_filter = df_aa_window_labeled.dropna(how="any")
        else:
            raise ValueError(f"An error has occurred, please check if the correct types have been inputted.")
        return cls(df_aa_window_labeled_filter)

    @classmethod
    def modify_label_by_ident_column(cls, df_label: pd.DataFrame, df_compare: pd.DataFrame, column_id: str,
                                     threshold: int = 2, weighting: bool = True):
        """
        Algorithm for changing labels, only for prior multi-annotation of protein sequences!
        Identification of matches based on given "column_aa_position" of "df_label", which must be contained in "df_compare"
        "column_aa_position" must be an int, otherwise it is disregarded!

        Parameters
        __________
        df_label : product of get_aa_window_df --> labelled window slices
        df_compare : pd.DataFrame that contains the name_label (compare get_aa_window_labels) of df_label in a column
        column_id : the "ID" index of df_label must be identical to the column of df_compare, required for filtering!
        threshold : required matches in df_compare slices (sliced by column_id entries), standard is 2
        weighting : creates a weight column depending on occurrence of window in data

        Returns
        _______
        df_label with modified labels
        """
        # identification list
        df_label_search_list = list(dict.fromkeys([str(index).split("__")[0] for index in df_label.index.tolist()]))
        # generate slices and iterate over df_label_search_list
        df_label_reset = df_label.reset_index()
        list_id = df_label_reset["ID"].to_numpy().tolist()
        df_compare_filtered = df_compare.dropna(subset=[column_id])
        position_seq_label = df_label.columns.tolist()[3]

        # threshold dependent, meaning that negative labels at thresh=3 need to be inverse
        if weighting:
            df_label["weights"] = np.zeros(df_label.shape[0], dtype=int).tolist()

        for query in df_label_search_list:
            # slice df_label
            # multilayer code for correct slice identification from df_label
            df_label_slice = df_label_reset[df_label_reset["ID"].str.contains(query)]
            arr_slice_positives = np.array([str(id_tag).split("__")[0] for id_tag in df_label_slice["ID"].tolist()])
            filter_list_slice_label = np.where(arr_slice_positives == query)
            df_label_slice = df_label_slice.iloc[filter_list_slice_label]

            list_available_pos = df_label_slice[position_seq_label].to_numpy().tolist()
            index_df_label_list = df_label_slice.index.tolist()

            # slice df_compare
            df_compare_slice_list_pos = (df_compare_filtered[df_compare_filtered[column_id].str.contains(query)]
                                         [position_seq_label].to_numpy().tolist())
            values, counts = np.unique(df_compare_slice_list_pos, return_counts=True)  # get pos in seq and their counts
            # the checking function
            for value, count in zip(values, counts):  # iterating over the pos in seq with their counts
                if count >= threshold:  # count of seq pos must be greater than the threshold
                    if value in list_available_pos:  # is the seq pos in the label list?
                        id_df_label = list_id[index_df_label_list[list_available_pos.index(int(value))]]
                        df_label.loc[id_df_label, "label"] = 1
                        if weighting:
                            df_label.loc[id_df_label, "weights"] = int(count)
                else:  # when negative label!
                    if value in list_available_pos:  # is the seq pos in the label list?
                        if weighting:
                            id_df_label = list_id[index_df_label_list[list_available_pos.index(int(value))]]
                            df_label.loc[id_df_label, "weights"] = int(df_label_slice.shape[0]-count)

        # remove all entries that are fully 0
        df_label_search_list = list(dict.fromkeys([str(index).split("__")[0] for index in df_label.index.tolist()]))
        for query in df_label_search_list:
            df_slice = get_df_slice(query, df_label).set_index("ID")
            if 1 not in df_slice["label"].to_numpy().tolist():
                df_label = df_label.drop(df_slice.index.tolist(), axis=0)
        return cls(df_label)

    # __________________________________________________________________________________________________________________
    # instance method to get a description of the DataFrames
    def window_describe(self):
        return label_describe(self.df)

    def show_df(self):
        return self.df
