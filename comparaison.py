# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:39:21 2020

@author: Louis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def open_csv(path):
    df = pd.read_csv(path)
    if 'year' in list(df.columns):
        df['year'] = pd.to_datetime(df['year'])
    return df


saved_songs = open_csv("datasets\output\mySavedSongs.csv")
data = open_csv("datasets\data.csv")

print(saved_songs.columns)
print(data.columns)

columns = list(set(saved_songs.columns) & set(data.columns)) 
cols = [col for col in columns if col not in ["id","name","key","mode","year"]]

print(cols)


#data[cols].hist(bins=10, figsize=(10,10), density=False)
#saved_songs[cols].hist(bins=10, figsize=(10,10), density=False)

fig, axes = plt.subplots(4, 3, figsize=(21, 18))

for i, col in enumerate(cols):
    ax = axes[round(i%3), round(i/4)]
    
    df = data[cols] #gapminder[gapminder.continent == 'Africa']
    sns.distplot(df[col], ax=ax, kde=False, label='data', norm_hist=True)
    
    df = saved_songs[cols] # =gapminder[gapminder.continent == 'Americas']
    sns.distplot(df[col],ax=ax,  kde=False,label='my songs', norm_hist=True)
    
    # Plot formatting
    ax.legend(prop={'size': 12})
    #ax.set_title(col)
   
    ax.set_ylabel('Density')

plt.show()
