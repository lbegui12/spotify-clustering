# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:59:52 2020

@author: Louis
"""
import hashlib
import pandas as pd
from helper import timer
from track import Track

class Playlist:
    def __init__(self):
        self.id = -1
        self.name = ""
        self.tracks = []
        self.count = 0
        
    def set_playlist_name(self, name):
        self.name = name
        hash_object = hashlib.sha1(name.encode())
        hex_dig = hash_object.hexdigest()
        print("Playlist '{}' - {}".format(name, hex_dig))
        self.id = hex_dig
        
    def add_track(self, track):
        self.tracks.append(track)
        self.count+=1
        
    def print_playlist(self):
        count=0
        for track in self.tracks:
            track.print_track()
            count+=1
        if self.count != count:
            self.count = count
            print("Ton count était pas bon...")
    
    @timer    
    def playlist_to_df(self):
        track_dicts = [track.to_dict() for track in self.tracks]
        return pd.DataFrame.from_records(track_dicts)
    
    def df_to_playlist(self, df, playlist_name = "playlist"):
        self.name = playlist_name
        for index, row in df.iterrows():
            t = Track(None)
            #print(row)
            t.id = row['id']
            t.artist = row['artists'][0]
            t.name = row['name']
            self.add_track(t)
        return self
       
        
        