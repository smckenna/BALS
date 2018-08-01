
# coding: utf-8

# %matplotlib inline

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import re
import csv

import os


# ### Individual Games
# 
# Read in data on individual games

# In[2]:

# make sure there's a date and csv
meta_re = re.compile('\_(\d{4})-(\d{2})-(\d{2})-(\d*).*(?=csv)', re.DOTALL)


# In[3]:

data_dir = '..\\..\\..\\videoStorage\\'
game_data = []

for file in os.listdir(data_dir):
    meta_info = set()
    meta_match = meta_re.search(file)
    
    if meta_match:
        meta_info = meta_match.groups()
        date = '-'.join(meta_info[:-1])
        g_id = meta_info[-1]
        
        with open(data_dir + file, 'rb') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if 'Duration' not in row:
                    full_row = row
                    full_row.append(row[((row[0] == row[2]) + 1) / 2])
                    game_data.append([date, g_id] + full_row)
            infile.close()
            


# In[4]:

game_data_cols = ['date', 'game_id', 'player1', 'player2', 'winner', 'stripes', 'solids', 'duration', 'loser']
df_games = pd.DataFrame(game_data, columns=game_data_cols)


# In[5]:

df_games['date'] = pd.to_datetime(df_games.date)
df_games['duration'] = df_games.duration.astype(float)
df_games['ind'] = 1


# In[6]:

df_games = df_games[((df_games.player1 != '') & (df_games.player2 != ''))]


# In[7]:

df_games


# ### Aggregate head-to-head records
# 
# Head to head records:

# In[8]:

games_h2h_ct = df_games[['winner', 'loser', 'ind']].groupby(['winner', 'loser']).sum().reset_index()
games_h2h = games_h2h_ct.pivot(index='loser', columns='winner', values='ind')

del games_h2h.columns.name
games_h2h.fillna(0, inplace=True)
games_h2h.reset_index(inplace=True)
games_h2h.set_index('loser', drop=True, inplace=True)
games_h2h.index.name = None

#games_h2h


# In[9]:

all_players = set(games_h2h_ct.winner.values).union(set(games_h2h_ct.loser.values))


# Adjust so that all players are included in both axes

# In[10]:

for player in all_players:
    winners = games_h2h.columns
    losers = games_h2h.index.values
    
    if player not in winners:
        games_h2h[player] = 0
    
    if player not in losers:
        games_h2h.loc[player] = list(np.zeros(games_h2h.shape[1]))


# Sort columns to make the matrix look nice

# In[11]:

games_h2h = games_h2h.reindex_axis(sorted(games_h2h.columns), axis=1)
games_h2h.sort_index(inplace=True)


# Add in total wins and losses for each player:

# In[12]:

games_h2h['losses'] = games_h2h.sum(axis=1)
games_h2h.loc['wins'] = list(games_h2h.sum(axis=0))[:-1] + [0]


# #### Head-to-head heatmap 
# 
# See [stackoverflow](http://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib) for borrowed code for labeling text. For some reason need to run this (and sorting below as well) twice in order to get correct index ordering.

# In[13]:

scale = 0.75
fig, ax = plt.subplots(figsize = (scale * games_h2h.shape[1], scale * games_h2h.shape[0]))
heatmap = ax.pcolor(games_h2h, cmap = "Blues")

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(games_h2h.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(games_h2h.shape[0])+0.5, minor=False)

ax.set_xticklabels(games_h2h.columns)
ax.set_yticklabels(games_h2h.index.values)

ax.invert_yaxis()
ax.xaxis.tick_top()

for y in range(games_h2h.shape[0]):
    for x in range(games_h2h.shape[1]):
        val = games_h2h.iloc[y, x]
        opp_val = games_h2h.iloc[x, y]
        if val or opp_val:
            plt.text(x + 0.5, y + 0.5, '%.1d' % val,
                     horizontalalignment='center',
                     verticalalignment='center',
                     )


# In[14]:

plt.show()


# In[15]:

print list(games_h2h.index.values)
print list(games_h2h.columns)


# #### Quick hits

# GZ stats

# In[16]:

gz_games = df_games[((df_games.player1 == 'GJZ') | (df_games.player2 == 'GJZ'))]


# In[17]:

print 'GZ win percentage: %.2f' % (len(gz_games[gz_games.winner == 'GJZ']) / float(len(gz_games)))


# How often does player who breaks win?

# In[18]:

print 'Player 1 (breaker) win percentage: %.2f' % (float(len(df_games[df_games.player1 == df_games.winner])) / len(df_games))


# In[ ]:




# ### Elo Rating Systems

# Elo parameters

# In[19]:

init_avg = 1500.
sel_k = 20.
exp_div = 400.

kbase = 800.
denom_const = 40. # higher values mean greater weighting on initial games


# #### Helper functions

# In[20]:

def p_win_exp(p1, p2, div=exp_div):
    p_win = 1. / (1. + 10. ** ((p2 - p1) / div))
    
    #print 'P(win)', '%.2f' % p_win 
    return p_win


# Simple $K$-factor

# In[21]:

def new_score(p1r, p2r, score=1, k=sel_k):
    # if p1 wins, score is 1
    # if p1 loses, score is -1
    delta = k * (score - p_win_exp(p1r, p2r))
    
    #print 'Point delta', '%.2f' % delta
    return p1r + delta, p2r - delta


# Adjusted $K$-factor

# In[22]:

def new_score_kadj1(p1r, p2r, g1, g2, kbase=kbase, score=1):
    # if p1 wins, score is 1
    # if p1 loses, score is -1
    const = (score - p_win_exp(p1r, p2r))
    delta1 = (kbase / (denom_const + g1)) * const
    delta2 = (kbase / (denom_const + g2)) * const
    
#     print delta1, g1
#     print delta2, g2

    return p1r + delta1, p2r - delta2


# #### Reset the Elo records

# In[23]:

seq_info = pd.DataFrame(games_h2h.index)
seq_info.columns = ['player']

seq_info['simple_elo'] = init_avg
seq_info['adj_elo1'] = init_avg
seq_info['num_wins'] = 0.
seq_info['tot_games'] = 0.
seq_info = seq_info[seq_info.player != 'wins']


# In[24]:

seq_info.set_index('player', inplace=True, drop=True)
seq_info.index.name = None


# In[25]:

assert((seq_info.simple_elo.values == 1500.).all())
assert((seq_info.adj_elo1.values == 1500.).all())


# #### Run through the game log:

# In[26]:

for idx, row in df_games.iterrows():
    winner = row['winner']
    loser = row['loser']
    
    g1 = seq_info.loc[winner]['tot_games']
    g2 = seq_info.loc[loser]['tot_games']
    
    # update game counts and victories
    seq_info.loc[winner]['num_wins'] += 1
    seq_info.loc[winner]['tot_games'] += 1
    seq_info.loc[loser]['tot_games'] += 1
    
    # adjust elo across all systems
    (seq_info.loc[winner]['simple_elo'], seq_info.loc[loser]['simple_elo'])         = new_score(seq_info.loc[winner]['simple_elo'], seq_info.loc[loser]['simple_elo'])

    (seq_info.loc[winner]['adj_elo1'], seq_info.loc[loser]['adj_elo1'])         = new_score_kadj1(seq_info.loc[winner]['adj_elo1'], seq_info.loc[loser]['adj_elo1'],
                         g1, g2)


# In[27]:

print seq_info.sort_values('simple_elo', ascending=False)


# In[28]:

sum(seq_info.simple_elo) / init_avg


# In[ ]:



