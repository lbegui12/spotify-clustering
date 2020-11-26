# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:26:06 2020

@author: Louis
"""


""" audio_features  : full doc : https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/
    
    key : The estimated overall key of the track. 
        Integers map to pitches using standard Pitch Class notation . 
        E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
    
    
    """
    
audio_features = ['danceability', 'energy', 'key', 'loudness', 
                'mode', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
features = ['name', 'id', 'artist', 'year', 'popularity']

class Track:
        
    def __init__(self, track):   
        if track is None:
            self.id = -1
            self.artist = ""
            self.name = ""
            self.year = 0
            self.popularity = 0
            self.audio_features = {}
        else:
            self.id = track['id']
            self.artist = track['artists'][0]['name']
            self.name = track['name']
            self.year = track['album']['release_date']
            self.popularity = track['popularity']
            self.audio_features = {}
        
    def to_dict(self):
        d = {
            'id' : self.id,
            'artist' : self.artist,
            'name' : self.name,
            'year' : self.year,
            'popularity': self.popularity
        }
        d.update(self.audio_features)
        return d
        
        
        
    def set_info(self, id, name, artist):
        self.id = id
        self.artist = artist
        self.name = name
        
    def set_features(self, audio_feat_dict):
        # Check key correspondent ?
        if all(elem in audio_feat_dict.keys() for elem in audio_features):
            for f in audio_features:
                self.audio_features[f] = audio_feat_dict[f]
        else:
            raise NameError("Audio feature missing key...")
                
    def print_track(self):
        print("'{}' by {}".format(self.name, self.artist))