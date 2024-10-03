import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.stats import skew, kurtosis, ttest_rel, ttest_1samp

# Load the dataset
df = pd.read_csv('C:/Users/Yash/Desktop/players_stats_by_season_full_details.csv')

# Filter the dataset to show only NBA regular season data
regular_season_df = df[df['Stage'] == 'Regular_Season']

# Determine the player who has played the most regular seasons
player_seasons = regular_season_df.groupby('Player')['Season'].nunique()
most_seasons_player = player_seasons.idxmax()

# Filter data for the player with the most regular seasons
player_data = regular_season_df[regular_season_df['Player'] == most_seasons_player]

# Calculate three-point accuracy for each season
player_data['3P_Accuracy'] = player_data['3PM'] / player_data['3PA']

# Perform linear regression for three-point accuracy across the years played
player_data['Season_Year'] = player_data['Season'].apply(lambda x: int(x.split(' - ')[0]))
X = player_data[['Season_Year']]
y = player_data['3P_Accuracy']
reg = LinearRegression().fit(X, y)

# Create a line of best fit
player_data['3P_Accuracy_Pred'] = reg.predict(X)

# Plot the data and the line of best fit
plt.scatter(player_data['Season_Year'], player_data['3P_Accuracy'], color='blue', label='Actual')
plt.plot(player_data['Season_Year'], player_data['3P_Accuracy_Pred'], color='red', label='Fit Line')
plt.xlabel('Season Year')
plt.ylabel('Three-Point Accuracy')
plt.title(f'Three-Point Accuracy Over Seasons for {most_seasons_player}')
plt.legend()
plt.show()

# Calculate the average three-point accuracy by integrating the fit line
def fit_line(x):
    return reg.coef_[0] * x + reg.intercept_

start_year = player_data['Season_Year'].min()
end_year = player_data['Season_Year'].max()
integral, _ = quad(fit_line, start_year, end_year)
average_3P_accuracy_fit = integral / (end_year - start_year)

# Calculate the actual average three-point accuracy
average_3P_accuracy_actual = player_data['3P_Accuracy'].mean()

# Interpolate missing values for the 2002-2003 and 2015-2016 seasons
missing_seasons = [2002, 2015]
interpolated_values = {season: fit_line(season) for season in missing_seasons}

# Output the results
print(f'Player with the most regular seasons: {most_seasons_player}')
print(f'Average three-point accuracy (fit line): {average_3P_accuracy_fit}')
print(f'Actual average three-point accuracy: {average_3P_accuracy_actual}')
for season, value in interpolated_values.items():
    print(f'Interpolated three-point accuracy for {season}-{season+1}: {value}')

# Calculate statistical measures for FGM and FGA
fgm_stats = {
    'mean': regular_season_df['FGM'].mean(),
    'variance': regular_season_df['FGM'].var(),
    'skew': skew(regular_season_df['FGM']),
    'kurtosis': kurtosis(regular_season_df['FGM'])
}

fga_stats = {
    'mean': regular_season_df['FGA'].mean(),
    'variance': regular_season_df['FGA'].var(),
    'skew': skew(regular_season_df['FGA']),
    'kurtosis': kurtosis(regular_season_df['FGA'])
}

# Print statistical measures
print("FGM Statistics:", fgm_stats)
print("FGA Statistics:", fga_stats)

# Perform a relational t-test on FGM and FGA
t_stat_rel, p_value_rel = ttest_rel(regular_season_df['FGM'], regular_season_df['FGA'])
print(f"\nRelational t-test between FGM and FGA: t-statistic: {t_stat_rel}, p-value: {p_value_rel}")

# Perform individual t-tests on FGM and FGA
t_stat_fgm, p_value_fgm = ttest_1samp(regular_season_df['FGM'], 0)
t_stat_fga, p_value_fga = ttest_1samp(regular_season_df['FGA'], 0)

print(f"\nIndividual t-test for FGM: t-statistic: {t_stat_fgm}, p-value: {p_value_fgm}")
print(f"\nIndividual t-test for FGA: t-statistic: {t_stat_fga}, p-value: {p_value_fga}")

# Compare the results of the individual t-tests to the relational t-test
print("\nComparison of t-tests:")
print(f"Relational t-test p-value: {p_value_rel}")
print(f"Individual t-test p-value for FGM: {p_value_fgm}")
print(f"Individual t-test p-value for FGA: {p_value_fga}")