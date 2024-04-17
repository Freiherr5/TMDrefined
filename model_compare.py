# standard util
import pandas as pd
import numpy as np
import glob
# visualize
import matplotlib.pyplot as plt
import seaborn as sns



def compare_predictions_heatmap(job_name: str, dict_dfs: dict, compare_column: str, allowed_index_list: list = None,
                                cmap: str = 'coolwarm'):

    if allowed_index_list is not None:
        for key, value in dict_dfs.items():
            index_intersect = [ind for ind in allowed_index_list if ind in value.index.tolist()]
            filtered_df = value.loc[index_intersect]
            dict_dfs[key] = filtered_df

    # generate the data
    data_df_list = []
    data_df_columns = ["tags"]
    for key_1, value_1 in dict_dfs.items():
        data_df_columns.append(key_1)
        row_slice = [key_1]
        for key_2, value_2 in dict_dfs.items():
            index_1 = value_1.index.tolist()
            index_2 = value_2.index.tolist()
            index_intersect = [ind for ind in index_1 if ind in index_2]
            list_1 = value_1.loc[index_intersect, compare_column].tolist()
            list_2 = value_2.loc[index_intersect, compare_column].tolist()
            counter_equal = 0
            len_dfs = len(list_1)
            for item_1, item_2 in zip(list_1, list_2):
                if item_1 == item_2:
                    counter_equal += 1
                else:
                    pass
            percent_match = counter_equal/len_dfs
            row_slice.append(percent_match)
        data_df_list.append(row_slice)
    data_df = pd.DataFrame(data_df_list, columns=data_df_columns).set_index("tags")


    # generate the mask for the heatmap
    count_df = len(dict_dfs)
    mask = []
    i = 1
    while True:
        row_mask_true = [False for pos in range(i)]
        row_mask_false = [True for pos in range(count_df-i)]
        row_mask_true.extend(row_mask_false)
        mask.append(row_mask_true)
        if i == count_df:
            break
        i += 1


    # heatmap generation
    # plotting a triangle correlation heatmap
    plt.rcParams["font.weight"] = "semibold"
    dataplot = sns.heatmap(data_df, annot=True, mask=np.array(mask, dtype=bool), center=0,
                           vmin=0, vmax=1, cmap=cmap, linewidths=3, linecolor='white',
                           fmt=".2f")
    plt.title(f"{job_name}\n", fontsize=14, fontweight="bold")
    # displaying heatmap
    plt.savefig(f"{job_name}.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    get_df_list_path = glob.glob("Heatmap_compare_non_sub/*.xlsx")
    list_df_heat = [pd.read_excel(df_path).set_index("entry") for df_path in get_df_list_path]
    print(get_df_list_path)
    make_dict_df = {"UniProt": list_df_heat[0],
                    "Membranome": list_df_heat[1],
                    "DeepTMHMM": list_df_heat[6],
                    "AM_no_weight_TMDrefined": list_df_heat[2],
                    "AM_weight_TMDrefined": list_df_heat[4],
                    "Expert_TMDrefined": list_df_heat[3],
                    "Expert_curated": list_df_heat[5]}

    uniprot_index = list_df_heat[0][list_df_heat[0]["dataset"] == "SUBLIT"].index.tolist()
    compare_predictions_heatmap(job_name="Percent of matching TMD stop-positions for SUBLIT", dict_dfs=make_dict_df,
                                compare_column="stop_pos_TMD", allowed_index_list=uniprot_index, cmap="BuPu")



    # SUBLIT = BuPu, SUBEXPERT = PuBu, NONSUB = YlOrRd







