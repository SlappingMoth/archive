# a test to predict pokemon types -


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as stats
from scipy.stats import ttest_ind

def main():
    ALPHA = .05

    pokemon = pd.read_csv("pokemon.csv", header=0, usecols=( 0, 3, 4, 5, 11, 21),
                          names=('number', 'name', 'type1', 'type2', 'gen', 'overallStat'), encoding='utf8')

    pokemon_number = pokemon['number']
    pokemon_name = pokemon['name']
    pokemon_type = pokemon['type1']
    pokemon_type2 = pokemon["type2"]
    pokemon_generation = pokemon['gen']
    pokemon_stat = pokemon['overallStat']

    #find all the pokemon that are fire type
    fire_pokemon = pokemon_name.where(pokemon_type == "Fire")
    fire_pokemon = fire_pokemon.dropna()
    fire_pokemon_number = pokemon_number.where(pokemon_type == "Fire")
    fire_pokemon_number = fire_pokemon_number.dropna()
    #find all the stats of the fire type pokemon found
    fire_pokemon_stats = pokemon_stat.where(pokemon_type == "Fire")
    fire_pokemon_stats = fire_pokemon_stats.dropna()
    fire_avg_stat = fire_pokemon_stats.mean()
    #put the name and corresponding stats together in a dictionary
    fire_dict = dict(zip(fire_pokemon, fire_pokemon_stats))

    dark_pokemon = pokemon_name.where(pokemon_type == "Dark")
    dark_pokemon = dark_pokemon.dropna()
    dark_pokemon_stats = pokemon_stat.where(pokemon_type == "Dark")
    dark_pokemon_stats = dark_pokemon_stats.dropna()
    dark_avg_stat = dark_pokemon_stats.mean()
    dark_dict = dict(zip(dark_pokemon, dark_pokemon_stats))

    #create an array to test the data with.
    x = np.concatenate((fire_pokemon_stats, dark_pokemon_stats))
    avg_x = x.mean()
    print("Stats: Skew:{:.2f}".format(stats.skew(x)))

    #run a shapiro test to test normality
    W, p = stats.shapiro(x)
    print("Shaprio-Wilk: W:{0} p={1}".format(W, p))
    if p < ALPHA:
        print("We can reject the hypothesis")
    else:
        print("We can not reject the hypothesis")

    #run a normal-test to test normality
    K2, p = stats.normaltest(x)
    print("Normal-test: k2:{0} p={1}".format(K2, p))
    if p < ALPHA:
        print("We can reject the hypothesis")
    else:
        print("We can not reject the hypothesis")

    #D, p = stats.kstest(x, "norm")
    #print("Kolmogorov-Smirnov: D:{0} p={1}".format(D, p))
    #if p < ALPHA:
        #print("We can reject the hypothesis")
    #else:
        #print("We can not reject the hypothesis")

    #stat, p = ttest_ind(fire_pokemon_stats, dark_pokemon_stats)
    #print('TTest: statistics:{0} p={1}'.format(stat, p))

    #run a mannwhitney u test
    stat, p = stats.mannwhitneyu(fire_pokemon_stats, dark_pokemon_stats)
    print("Mannwhitneyu: statistics:{0} p={1}".format(stat, p))
    if p < ALPHA:
        print("We can reject the hypothesis")
    else:
        print("We can not reject the hypothesis")

    plot_data(avg_x)

def plot_data(x):
    #create the histogram to check the normality of the distribution
    plt.hist(x, bins=8, edgecolor="black", color="teal", label="Fire and Dark Pokemon")
    plt.axvline(x=x, color="blue", label="Pokemon Mean")
    #plt.hist(fire_pokemon_stats, bins=8, edgecolor="black", color="red", label="Fire Pokemon")
    #plt.axvline(x=fire_avg_stat, color="maroon", label="Fire mean")
    #plt.hist(dark_pokemon_stats, bins=8, edgecolor="black", color='purple', label="Dark Pokemon")
    #plt.axvline(x=dark_avg_stat, color="gray", label="Dark mean")
    plt.legend()
    plt.xscale("linear")
    plt.yscale("linear")
    plt.title("Fire vs Dark type Pokemon in term of Overall Stats")
    plt.xlabel("The overall stats of the Pokemon")
    plt.ylabel("The Distribution of fire and Dark types")
    plt.show()




if __name__ == "__main__":

    main()

