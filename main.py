import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from daria import DARIA


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 2, 2)
    stacked = True
    width = 0.6
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
        ncol = 2
    else:
        ncol = 5
    
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (10,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results/bar_chart_' + title[-4:] + '.pdf')
    plt.show()



# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (8, 5))
    sns.set(font_scale = 1.2)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".2f", cmap="PRGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '.pdf')
    plt.show()


def main():
    
    path = 'dataset'
    # Number of countries
    m = 27

    # Symbols of Countries
    coun_names = pd.read_csv('dataset/country_symbols.csv')
    country_names = list(coun_names['Symbol'])

    str_years = [str(y) for y in range(2015, 2021)]
    # dataframe for annual results TOPSIS
    preferences_t = pd.DataFrame(index = country_names)
    rankings_t = pd.DataFrame(index = country_names)

    # dataframe for annual results VIKOR
    preferences_v = pd.DataFrame(index = country_names)
    rankings_v = pd.DataFrame(index = country_names)

    mat_avg = np.zeros((27, 9))
    # initialization of TOPSIS
    topsis = TOPSIS()
    # initialization of VIKOR
    vikor = VIKOR()

    # dataframes for results summary
    pref_summary = pd.DataFrame(index = country_names)
    rank_summary = pd.DataFrame(index = country_names)

    for el, year in enumerate(str_years):
        file = 'data_sdg11_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        # types: 1 profit -1 cost
        types = np.array([-1, -1, 1, -1, -1, 1, 1, 1, -1])
        
        list_of_cols = list(data.columns)
        # matrix
        matrix = data.to_numpy()
        # weights Entropy
        weights = mcda_weights.entropy_weighting(matrix)

        # TOPSIS annual
        pref_t = topsis(matrix, weights, types)
        rank_t = rank_preferences(pref_t, reverse = True)
        
        preferences_t[year] = pref_t
        rankings_t[year] = rank_t

        # VIKOR annual
        pref_v = vikor(matrix, weights, types)
        rank_v = rank_preferences(pref_v, reverse=False)

        preferences_v[year] = pref_v
        rankings_v[year] = rank_v

        if year == '2020':
            pref_summary['2020 TOPSIS'] = pref_t
            pref_summary['2020 VIKOR'] = pref_v

            rank_summary['2020 TOPSIS'] = rank_t
            rank_summary['2020 VIKOR'] = rank_v

        mat_avg += matrix


    preferences_t.to_csv('results/preferences_t.csv')
    preferences_v.to_csv('results/preferences_v.csv')
    
    rankings_t.to_csv('results/rankings_t.csv')
    rankings_v.to_csv('results/rankings_v.csv')
    # Create dataframes for summary of preferences and rankings
    
    # AVG performances of alternatives
    mat_avg = mat_avg / len(str_years)
    # weights Entropy
    weights_avg = mcda_weights.entropy_weighting(mat_avg)

    # TOPSIS AVG
    pref_avg_t = topsis(mat_avg, weights_avg, types)
    rank_avg_t = rank_preferences(pref_avg_t, reverse=True)

    # VIKOR AVG
    pref_avg_v = vikor(mat_avg, weights_avg, types)
    rank_avg_v = rank_preferences(pref_avg_v, reverse=False)

    pref_summary['AVG TOPSIS'] = pref_avg_t
    rank_summary['AVG TOPSIS'] = rank_avg_t

    pref_summary['AVG VIKOR'] = pref_avg_v
    rank_summary['AVG VIKOR'] = rank_avg_v

    # saved temporal matrix with annual results
    
    # PLOT  TOPSIS =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (7, 6))
    for i in range(rankings_t.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_t.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max - 0.15, rankings_t.iloc[i, -1]),
                        fontsize = 12, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, str_years, fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('TOPSIS Rankings')
    plt.tight_layout()
    plt.savefig('results/rankings_years_t' + '.pdf')
    plt.show()
    
    # PLOT  VIKOR =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (7, 6))
    for i in range(rankings_v.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_v.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max - 0.15, rankings_v.iloc[i, -1]),
                        fontsize = 12, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    plt.xticks(x1, str_years, fontsize = 12)
    plt.yticks(ticks, fontsize = 12)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('VIKOR Rankings')
    plt.tight_layout()
    plt.savefig('results/rankings_years_v' + '.pdf')
    plt.show()

    

    # ======================================================================
    # DARIA-TOPSIS - 1
    # ======================================================================
    # DARIA (DAta vaRIAbility) temporal approach
    # preferences includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = country_names)
    df = preferences_t.T
    matrix = df.to_numpy()

    # TOPSIS orders preferences in descending order
    met = 'topsis'
    type = 1

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values with Gini coefficient
    var = daria._gini(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, type)

    # for next stage of research
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_results = pd.DataFrame()
    df_results['Ai'] = list(df.columns)
    df_results['Variability'] = list(var)
    
    # list of directions
    df_results['dir list'] = dir_list
    
    df_results.to_csv('results/scores_t.csv')
    df_varia_fin = df_varia_fin.rename_axis('Ai')
    df_varia_fin.to_csv('results/FINAL_T.csv')

    # final calculation
    # data with alternatives' rankings' variability values calculated with Gini coeff and directions
    G_df = copy.deepcopy(df_varia_fin)

    # data with alternatives' efficiency of performance calculated for the recent period
    S_df = copy.deepcopy(preferences_t)

    # ==============================================================
    S = S_df['2020'].to_numpy()

    G = G_df[met.upper()].to_numpy()
    dir = G_df[met.upper() + ' dir'].to_numpy()

    # update efficiencies using DARIA methodology
    # final updated preferences
    final_S = daria._update_efficiency(S, G, dir)

    # TOPSIS has descending ranking from prefs
    rank = rank_preferences(final_S, reverse = True)

    pref_summary['DARIA-TOPSIS'] = final_S
    rank_summary['DARIA-TOPSIS'] = rank

    # ======================================================================
    # DARIA-VIKOR - 2
    # ======================================================================
    # DARIA (DAta vaRIAbility) temporal approach
    # preferences includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = country_names)
    df = preferences_v.T
    matrix = df.to_numpy()

    # VIKOR orders preferences in ascending order
    met = 'vikor'
    type = -1

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values with Gini coefficient
    var = daria._gini(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, type)

    # for next stage of research
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_results = pd.DataFrame()
    df_results['Ai'] = list(df.columns)
    df_results['Variability'] = list(var)
    
    # list of directions
    df_results['dir list'] = dir_list
    
    df_results.to_csv('results/scores_v.csv')
    df_varia_fin = df_varia_fin.rename_axis('Ai')
    df_varia_fin.to_csv('results/FINAL_V.csv')

    # final calculation
    # data with alternatives' rankings' variability values calculated with Gini coeff and directions
    G_df = copy.deepcopy(df_varia_fin)

    # data with alternatives' efficiency of performance calculated for the recent period
    S_df = copy.deepcopy(preferences_v)

    # ==============================================================
    S = S_df['2020'].to_numpy()

    G = G_df[met.upper()].to_numpy()
    dir = G_df[met.upper() + ' dir'].to_numpy()

    # update efficiencies using DARIA methodology
    # final updated preferences
    final_S = daria._update_efficiency(S, G, dir)

    # VIKOR has ascending ranking from prefs
    rank = rank_preferences(final_S, reverse = False)

    pref_summary['DARIA-VIKOR'] = final_S
    rank_summary['DARIA-VIKOR'] = rank


    # =====================================================================
    # saving whole results
    pref_summary = pref_summary.rename_axis('Country')
    rank_summary = rank_summary.rename_axis('Country')
    pref_summary.to_csv('results/pref_summary.csv')
    rank_summary.to_csv('results/rank_summary.csv')



    # correlations for PLOT
    method_types = list(rank_summary.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_rs = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rank_summary[i], rank_summary[j]))
            dict_new_heatmap_rs[j].append(corrs.spearman(rank_summary[i], rank_summary[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')

    # correlation matrix with rs coefficient
    draw_heatmap(df_new_heatmap_rs, r'$r_s$')


if __name__ == '__main__':
    main()