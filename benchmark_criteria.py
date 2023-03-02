import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyrepo_mcda import weighting_methods as mcda_weights

from pyrepo_mcda.mcda_methods import TOPSIS, VIKOR

from pyrepo_mcda import normalizations as norms

from pyrepo_mcda import correlations as corrs

from pyrepo_mcda.additions import rank_preferences
import itertools
from daria import DARIA
import copy

import seaborn as sns


def main():
    
    iterations = 1000

    # criteria
    crits = np.array([6, 8, 10, 16, 24, 32])
    crits_ax = np.arange(2, 14, 2)
    
    # standard number of criteria
    n = 9

    # sstandard number of alternatives
    m = 27

    # standard number of periods
    periods = 6

    x1 = []
    x = []
    y = []
    w = []
    df_final = pd.DataFrame()


    list_local = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
    
    topsis = TOPSIS()
    vikor = VIKOR()

    # main loop with number of iterations
    for i in range(iterations):
        # loop with Criteria
        for it, n in enumerate(crits):

            mat_avg = np.zeros((m, n))

            list_of_matrices = []
            # loop with number of periods we consider
            for p in range(periods):
                matrix = np.random.uniform(1, 100, size = (m, n))
                list_of_matrices.append(matrix)

            # DARIA-TOPSIS
            preferences = pd.DataFrame()

            # save annual rankings - annual matrix with TOPSIS scores
            
            for p, matrix in enumerate(list_of_matrices):
                types = np.ones(matrix.shape[1])
                weights = mcda_weights.entropy_weighting(matrix)

                pref = topsis(matrix, weights, types)
                preferences[str(p + 1)] = pref

                mat_avg += matrix


            # TOPSIS AVG
            # AVG performances of alternatives
            mat_avg = mat_avg / len(list_of_matrices)
            # weights Entropy
            weights_avg = mcda_weights.entropy_weighting(mat_avg)

            # TOPSIS AVG
            pref_avg_t = topsis(mat_avg, weights_avg, types)
            rank3 = rank_preferences(pref_avg_t, reverse=True)

            fun_name3 = 'AVG TOPSIS'

            # VIKOR AVG
            pref_avg_v = vikor(mat_avg, weights_avg, types)
            rank4 = rank_preferences(pref_avg_v, reverse=False)

            fun_name4 = 'AVG VIKOR'

            
            # DARIA - most recent
            # ======================================================================
            # DARIA (DAta vaRIAbility) temporal approach
            # preferences includes preferences of alternatives for evaluated years
            df_varia_fin = pd.DataFrame()
            df = preferences.T
            matrix = df.to_numpy()

            # TOPSIS orders preferences in descending order
            met = 'topsis'
            type = 1

            # calculate efficiencies variability using DARIA methodology
            daria = DARIA()
            # calculate variability values with Entropy
            var = daria._gini(matrix)
            # calculate variability directions
            dir_list, dir_class = daria._direction(matrix, type)

            # for next stage of research
            df_varia_fin[met.upper()] = list(var)
            df_varia_fin[met.upper() + ' dir'] = list(dir_class)

            df_varia_fin = df_varia_fin.rename_axis('Ai')

            # final calculation
            # data with alternatives' rankings' variability values calculated with Gini coeff and directions
            G_df = copy.deepcopy(df_varia_fin)

            # data with alternatives' efficiency of performance calculated for the recent period
            S_df = copy.deepcopy(preferences)
            
            # most recent year updated by variability
            S = S_df.iloc[:, -1].to_numpy()

            G = G_df[met.upper()].to_numpy()
            dir = G_df[met.upper() + ' dir'].to_numpy()

            # update efficiencies using DARIA methodology
            # final updated preferences
            final_S = daria._update_efficiency(S, G, dir)

            # TOPSIS has descending ranking from prefs
            rank1 = rank_preferences(final_S, reverse = True)

            fun_name1 = 'DARIA-TOPSIS'


            # =============================================================
            # DARIA-VIKOR
            preferences = pd.DataFrame()

            # save annual rankings - annual matrix with VIKOR scores
            
            for p, matrix in enumerate(list_of_matrices):
                types = np.ones(matrix.shape[1])
                weights = mcda_weights.entropy_weighting(matrix)

                pref = vikor(matrix, weights, types)
                preferences[str(p + 1)] = pref

            
            # DARIA - most recent
            # ======================================================================
            # DARIA (DAta vaRIAbility) temporal approach
            # preferences includes preferences of alternatives for evaluated years
            df_varia_fin = pd.DataFrame()
            df = preferences.T
            matrix = df.to_numpy()

            # VIKOR orders preferences in ascending order
            met = 'vikor'
            type = -1

            # calculate efficiencies variability using DARIA methodology
            daria = DARIA()
            # calculate variability values with Entropy
            var = daria._gini(matrix)
            # calculate variability directions
            dir_list, dir_class = daria._direction(matrix, type)

            # for next stage of research
            df_varia_fin[met.upper()] = list(var)
            df_varia_fin[met.upper() + ' dir'] = list(dir_class)

            df_varia_fin = df_varia_fin.rename_axis('Ai')

            # final calculation
            # data with alternatives' rankings' variability values calculated with Gini coeff and directions
            G_df = copy.deepcopy(df_varia_fin)

            # data with alternatives' efficiency of performance calculated for the recent period
            S_df = copy.deepcopy(preferences)
            
            # most recent year updated by variability
            S = S_df.iloc[:, -1].to_numpy()

            G = G_df[met.upper()].to_numpy()
            dir = G_df[met.upper() + ' dir'].to_numpy()

            # update efficiencies using DARIA methodology
            # final updated preferences
            final_S = daria._update_efficiency(S, G, dir)

            # VIKOR has ascending ranking from prefs
            rank2 = rank_preferences(final_S, reverse = False)

            fun_name2 = 'DARIA-VIKOR'

            # AVERAGE

        
            # DARIA-TOPSIS / DARIA-VIKOR
            corr = corrs.weighted_spearman(rank1, rank2)
            y.append(corr)
            x.append(crits_ax[it] + list_local[1])
            x1.append(crits[it])
            w.append(fun_name1 + '/' + fun_name2)

            # DARIA-TOPSIS / AVG TOPSIS
            corr = corrs.weighted_spearman(rank1, rank3)
            y.append(corr)
            x.append(crits_ax[it] + list_local[2])
            x1.append(crits[it])
            w.append(fun_name1 + '/' + fun_name3)

            # DARIA-VIKOR / AVG VIKOR
            corr = corrs.weighted_spearman(rank2, rank4)
            y.append(corr)
            x.append(crits_ax[it] + list_local[3])
            x1.append(crits[it])
            w.append(fun_name2 + '/' + fun_name4)

            # AVG TOPSIS / AVG VIKOR
            corr = corrs.weighted_spearman(rank3, rank4)
            y.append(corr)
            x.append(crits_ax[it] + list_local[4])
            x1.append(crits[it])
            w.append(fun_name3 + '/' + fun_name4)

            
    # Criteria Alternatives
    df_final['Criteria'] = x1
    df_final['Correlation'] = y
    df_final['Methods'] = w
    df_final.to_csv('benchmarkable/data_crits.csv')
    df_final = df_final.rename_axis('Lp')
    df_final.to_csv('benchmarkable/dataLp_crits.csv')


    # plot visualization
    sns.set_style("darkgrid")
    plt.figure(figsize = (11, 5))
    ax = sns.boxenplot(x = 'Criteria', y = 'Correlation', hue = 'Methods', data = df_final)

    ax.tick_params(axis='x')
    ax.set_xlabel('Number of criteria')
    ax.set_ylabel(r'$r_w$' + ' correlation coefficient')
    ax.grid(True)
    ax.set_axisbelow(True)
    plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc='center',
    ncol=4, borderaxespad=0., edgecolor = 'black', title = 'Methods compared')
    plt.tight_layout()
    plt.savefig('benchmarkable/boxenplot_crits.pdf')
    plt.show()

    


if __name__ == '__main__':
    main()