import glob
import pandas as pd
import os
import numpy as np
import csv
import trueskill as ts
import datetime
from matplotlib import pyplot as plt

class Player:

    def __init__(self):
        self.initials     = ''
        self.games        = 0
        self.wins         = 0
        self.losses       = 0
        self.solids       = 0
        self.stripes      = 0
        self.meanDuration = np.nan
        self.minDuration  = np.nan
        self.maxDuration  = np.nan
        self.broke        = np.nan
        self.ranking      = np.nan
        
def fix_initials(df, old, new):
    df['Player1'].replace(old, new, inplace=True)
    df['Player2'].replace(old, new, inplace=True)
    df['Winner'].replace( old, new, inplace=True)
    df['Solids'].replace( old, new, inplace=True)
    df['Stripes'].replace(old, new, inplace=True)
    
    return df
    

def build_html(head, date, body, facts, tail, outFile):

    with open(head,'r') as f:
        newlines = []
        for line in f.readlines():
            newlines.append(line)

    with open(date,'r') as f:
        for line in f.readlines():
            newlines.append(line)
            
    with open(body,'r') as f:
        for line in f.readlines():
            newlines.append(line)

    with open(facts,'r') as f:
        for line in f.readlines():
            newlines.append(line)

    with open(tail,'r') as f:
        for line in f.readlines():
            newlines.append(line)
    
    with open(outFile, 'w') as f:
        for line in newlines:
            f.write(line)


def apply_custom_css(inputFile, cssFile):

    with open(cssFile,'r') as f:
        newlines = []
        for line in f.readlines():
            newlines.append(line)
    
    with open(inputFile,'r') as f:
        for line in f.readlines():
            newlines.append(line.replace('class="dataframe"', 'class="crispTable"'))
    
    with open(inputFile, 'w') as f:
        for line in newlines:
            f.write(line)


def fix_table_code(inputFile):

    with open(inputFile, 'r') as f:
        newlines = []
        for line in f.readlines():
            newlines.append(line.replace('class="dataframe"', ''))
    
    f.close()
    
    with open(inputFile, 'w') as f:
        for line in newlines:
            f.write(line)

def generate_h2h_heatmap(df_games):
  
    df_games['ind'] = 1
    
    games_h2h_ct = df_games[['Winner', 'Loser', 'ind']].groupby(['Winner', 'Loser']).sum().reset_index()
    games_h2h = games_h2h_ct.pivot(index='Loser', columns='Winner', values='ind')
    
    del games_h2h.columns.name
    games_h2h.fillna(0, inplace=True)
    games_h2h.reset_index(inplace=True)
    games_h2h.set_index('Loser', drop=True, inplace=True)
    games_h2h.index.name = None
    
    all_players = set(games_h2h_ct.Winner.values).union(set(games_h2h_ct.Loser.values))
    
    # Adjust so that all players are included in both axes
    for player in all_players:
        winners = games_h2h.columns
        losers  = games_h2h.index.values
        
        if player not in winners:
            games_h2h[player] = 0
        
        if player not in losers:
            games_h2h.loc[player] = list(np.zeros(games_h2h.shape[1]))
    
    # Sort columns to make the matrix look nice
    games_h2h = games_h2h.reindex_axis(sorted(games_h2h.columns), axis=1)
    games_h2h.sort_index(inplace=True)
    
    # Add in total wins and losses for each player:
    games_h2h['losses']   = games_h2h.sum(axis=1)
    games_h2h.loc['wins'] = list(games_h2h.sum(axis=0))[:-1] + [0]
    
    foo = games_h2h.sum(axis=0) + games_h2h.sum(axis=1)
    games_h2h_subset = games_h2h.loc[foo>=5, foo>=5]
    
    # Head-to-head heatmap 
    # See [stackoverflow](http://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib) for borrowed code for labeling text. For some reason need to run this (and sorting below as well) twice in order to get correct index ordering.
    scale = 0.75
    fig, ax = plt.subplots(figsize = (scale * games_h2h_subset.shape[1], scale * games_h2h_subset.shape[0]))
    heatmap = ax.pcolor(games_h2h_subset, cmap = "YlGn")
    
    # Put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(games_h2h_subset.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(games_h2h_subset.shape[0])+0.5, minor=False)
    
    ax.set_xticklabels(games_h2h_subset.columns)
    ax.set_yticklabels(games_h2h_subset.index.values)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    for y in range(games_h2h_subset.shape[0]):
        for x in range(games_h2h_subset.shape[1]):
            val = games_h2h_subset.iloc[y, x]
            opp_val = games_h2h_subset.iloc[x, y]
            if val or opp_val:
                if val > 0:
                    plt.text(x + 0.5, y + 0.5, '%.1d' % val,
                             horizontalalignment='center',
                             verticalalignment  ='center',
                             fontsize=16)
    
#    plt.show()
    fig.savefig('h2h_heatmap.jpg', dpi=300, transparent=True, bbox_inches='tight')
    
    return fig

    
if os.name == 'nt':
    foo = 1
else:
    os.chdir('/home/smckenna/BALS/code')
   

ELOMEAN = 25.0

files = list(np.sort(glob.glob('../videoStorage/BALS_2018*meta.csv')))

df = pd.DataFrame()
for file in files:
    df_ = pd.read_csv(file)
    df = df.append(df_)
    df['Loser'] = ""

df = df.dropna()
avgGameLength = np.mean(df.Duration)
totalNumberGames = len(df)
stripesWin = 0.0
breakWins  = 0.0

df['gameID'] = range(1, len(df) + 1)    
df.set_index('gameID', inplace=True)

df = fix_initials(df, 'RAL', 'RL')
df = fix_initials(df, 'REL', 'RL')
df = fix_initials(df, 'NS', 'NAS')

p1 = df['Player1'].unique()
p2 = df['Player2'].unique()
playerList = np.unique(np.concatenate((p1,p2)))

ts.setup(mu=ELOMEAN, sigma=(ELOMEAN/3)**2, draw_probability=0.00)
rankings = {p: ts.Rating() for p in playerList}
trends   = {p:[] for p in playerList}

for g in range(len(df)):
    won = df.Winner[g+1]
    if df.Player1[g+1] == won:
        lost = df.Player2[g+1]
        winner = rankings[won]                
        loser = rankings[lost]                
        newP1, newP2 = ts.rate_1vs1(winner, loser)
        rankings[won]  = newP1
        rankings[lost] = newP2
        if df.Stripes[g+1] == won:
           stripesWin = stripesWin + 1.0
        breakWins = breakWins + 1.0
        df.loc[g+1,'Loser'] = df.loc[g+1,'Player2']
    else:
        lost = df.Player1[g+1]
        winner = rankings[won]                
        loser = rankings[lost]                
        newP1, newP2 = ts.rate_1vs1(winner, loser)
        rankings[won]  = newP1
        rankings[lost] = newP2
        if df.Stripes[g+1] == won:
           stripesWin = stripesWin + 1.0
        trends[won].append(rankings[won])
        trends[lost].append(rankings[lost])
        df.loc[g+1,'Loser'] = df.loc[g+1,'Player1']

players   = []
standings = pd.DataFrame()

for p in playerList:
    player = Player()
    player.initials = p

    g1 = df['Player1'].value_counts()
    g2 = df['Player2'].value_counts()
    player.games = 0
    if p in g1: 
        player.games = g1[p]
    if p in g2: 
        player.games = player.games + g2[p]
    
    player.wins = 0
    w = df['Winner'].value_counts()
    if p in w:
        player.wins = w[p]
        
    player.losses = player.games - player.wins

    player.solids = 0
    s = df['Solids'].value_counts()
    if p in s:
        player.solids = s[p]

    player.stripes = player.games - player.solids

    d = df[(df.Player1 == p) | (df.Player2 == p)]
    player.meanDuration = np.round(np.mean(d.Duration),2)
    player.minDuration  = np.round(np.min( d.Duration),2)
    player.maxDuration  = np.round(np.max( d.Duration),2)
    
    player.broke = 0
    if p in g1: 
        player.broke = g1[p]
    
    player.ranking = rankings[p]
    
    players.append(player)
    
    if player.games >= 2:
        pct = 1.0*player.wins/(1.0*player.games)
        df_ = pd.DataFrame({'Player': player.initials, 'Wins': player.wins, 
        'Losses': player.losses, 'Pct': pct, 'Mean': player.ranking.mu,
        'Sigma': player.ranking.sigma}, index=[0])
        standings = standings.append(df_)
     
truSkill    = standings.Mean - 3.0*standings.Sigma
leaderBoard = standings
leaderBoard['TruSkill'] = truSkill

leaderBoard = leaderBoard.sort_values(by='TruSkill', ascending=False)
leaderBoard = leaderBoard[['Player', 'Wins', 'Losses', 'Pct', 'TruSkill']]
leaderBoard.to_csv('standings.csv'       , float_format='%04.3f', index=False, header=True)
leaderBoard.to_html('standingsTable.html', float_format='%04.3f', index=False, header=True)

fix_table_code('standingsTable.html')

with open('date.txt', 'w') as f:
    f.write(datetime.datetime.now().strftime('As of: %b %d, %Y at %H:%M'))
    f.write('</h3><br>')
with open('facts.txt', 'w') as f:
    f.write('<br><b>Win rate for Stripes: %.1f%%</b><br>\n'    % (100.0*stripesWin/totalNumberGames))        
    f.write('<b>Win rate for Break:   %.1f%%</b><br>\n'        % (100.0*breakWins/totalNumberGames))
    f.write('<b>Average game duration: %.2f minutes</b><br>\n' % avgGameLength)

build_html('head.html', 'date.txt', 'standingsTable.html', 'facts.txt', 'tail.html', 'standings.html')

heatmap = generate_h2h_heatmap(df)
