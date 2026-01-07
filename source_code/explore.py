

#all the imports into the program
import numpy as np
import matplotlib.pyplot as plt

def main():
    batting = np.genfromtxt("BattingStats.csv", dtype=None, delimiter=",", skip_header=1,
                              usecols=(0, 1, 3, 4, 6, 8, 11, 16),
                            names=('player', 'date', 'team', 'league', 'atBats', 'hits', "hr", 'SO'),
                            encoding='utf8', converters={"hits": num_zero, "hr": num_zero, "SO": num_zero})
    #set all the colomns used into variables
    players = batting['player']
    years = batting['date']
    teams = batting['team']
    league = batting['league']
    at_bats = batting['atBats']
    season_hits = batting['hits']
    home_runs = batting['hr']
    strike_out = batting['SO']

    #find the max index on th basis of most hits in one season
    max_index_season_hits = np.nanargmax(season_hits)
    max_hits = season_hits[max_index_season_hits]
    max_at_bats = season_hits[max_index_season_hits]# this find the max
    player_with_most_hits = players[max_index_season_hits]#find the player with the most hits in one season

    #find the min index based on the number of hits
    min_index_season_hits = np.nanargmin(season_hits)
    min_hits = season_hits[min_index_season_hits]
    min_at_bats = season_hits[min_index_season_hits]#this finds the player with the least at bats
    player_with_least_hits = players[min_index_season_hits]# this find the player with the least amount of hits

    max_index_home_runs = np.nanargmax(home_runs)
    max_home_runs = home_runs[max_index_home_runs]
    player_with_most_hr = players[max_index_home_runs]

    min_index_home_runs = np.nanargmin(home_runs)
    min_home_runs = home_runs[min_index_home_runs]
    player_with_least_hr = players[min_index_home_runs]

    max_index_strike_out = np.nanargmax(strike_out)
    max_strike_outs = strike_out[max_index_strike_out]
    player_with_most_strike_outs = players[max_index_strike_out]
    min_index_strike_out = np.nanargmin(strike_out)
    min_strike_outs = strike_out[min_index_strike_out]
    player_with_least_strike_outs = players[max_index_strike_out]
    mean_strike_out = np.nanmean(strike_out)



    #this is finding the best player in terms of hits
    first_player = np.argwhere(players == 'rosepe01')#ichiro has the most hits in one season- aswell as the most atbats in one season.
    first_player_avg = np.nanmean(strike_out[first_player])
    first_years_array = np.array(years[first_player])#find all the years that the best player played in the MLB
    first_hits_array = np.array(season_hits[first_player])

    #find a second player to plot
    second_player = np.argwhere(players == 'troutmi01')#mike trout of the Angles
    second_player_avg = np.nanmean(strike_out[second_player])
    second_years_array = np.array(years[second_player])
    second_hits_array = np.array(season_hits[second_player])
    #third_player = np.argwhere(players == 'ripkenjrca01')


    #grab a specific team to plot
    angels_team = np.argwhere(teams == 'LAA')# the team of choice is the angels
    angels_years = np.argwhere(years == 2002)
    angels_hits = season_hits[angels_team]# grab the hits of the angels
    angels_hits_avg = np.nanmean(angels_hits)
    angels_home_runs = home_runs[angels_team]# grab the homeruns of the angels


    #grab only the american league

    national_league = np.argwhere(league == "NL")
    american_league = np.argwhere(league == "AL")
    american_league_strike_outs = np.nanmean(strike_out[american_league])
    national_league_strike_outs = np.nanmean(strike_out[national_league])
    mlb_strike_out_avg = np.nanmean(strike_out)
    third_player = np.argwhere(players == "hollima01")
    third_player_avg = np.nanmean(strike_out[third_player])
    fourth_player = np.argwhere(players == "pujolal01")
    fourth_player_avg = np.nanmean(strike_out[fourth_player])
    fifth_player = np.argwhere(players == 'ripkeca01')
    fifth_player_avg = np.nanmean(strike_out[fifth_player])


    strike_out_list = [first_player_avg, second_player_avg, third_player_avg, fourth_player_avg, fifth_player_avg]
    labels = ["Pete Rose", "Mike Trout", "Matt Holiday", "Albert Pujols", "Cal Ripken Jr."]

    print("Player with most hits in a single season is", player_with_most_hits, "with", max_hits, "hits.")
    print("Player with the least hits in a single season is", player_with_least_hits, "with", min_hits, "hits.")
    print("Player with the most home runs in a single season is", player_with_most_hr, "with", max_home_runs, "home runs." )
    print("Player with the least home runs in a single season is", player_with_least_hr, "with", min_home_runs, "home runs.")
    print("Player with the most strike outs in a single season is", player_with_most_strike_outs, "with", max_strike_outs, "strike outs.")
    print("Player with the least strike outs in a single season is", player_with_least_strike_outs, "with", min_strike_outs, "strike outs.")

    print("The Average hits the Angels achieve in a single season is", "{:.2f}".format(angels_hits_avg))

    #plot a graph for Pete Rose
    bins = first_years_array.flatten()
    #this is the code to plot the data of Ichrio Suzuki/ using a histogram
    plt.hist(first_years_array, bins=bins, edgecolor='black', weights=first_hits_array)
    plt.title("Pete Rose's Total Hits per Season")
    plt.xlabel("Year of Baseball Season")
    plt.ylabel("Number of Hits per Season")
    plt.show()

    #create a bar graph for mike trout
    bins = second_years_array.flatten()
    plt.hist(second_years_array, bins=bins, edgecolor='black', weights=second_hits_array)
    plt.title("Mike Trout's Total hits per Season")
    plt.xlabel("Year of Baseball Season")
    plt.ylabel("Number of Hits per Season")
    plt.show()



    colors = [7, 5, 9]

    plt.scatter(x=angels_hits, y=angels_home_runs, s=1, c='red', cmap="blue" )
    plt.title("Angels: Correlation between hits and Home Runs")
    plt.xlabel("Hits per Season")
    plt.ylabel("HomeRuns per Season")
    plt.show()


    plt.pie(strike_out_list, labels=labels)
    plt.title("Comparing Players Strike outs")
    plt.show()


def num_zero(num_str):
    try:
        num = float(num_str)
    except ValueError:
        num = np.nan
    if num <= 0:
        num = np.nan
    return num



if __name__ == "__main__":
    main()