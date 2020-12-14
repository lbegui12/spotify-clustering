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
    
audio_features = ['danceability',       # Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
                  'energy',             # Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. 
                  'loudness',           # The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
                  'speechiness',        # Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 
                  'acousticness',       # A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
                  'instrumentalness',   # Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. 
                  'liveness',           # Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
                  'valence',            # A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
                  'tempo',              # The overall estimated tempo of a track in beats per minute (BPM).
                  'duration_ms',        # The duration of the track in milliseconds.
                  'key',                # The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
                  'mode',               # Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
                  'time_signature'      # An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
                  ]

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
            'artists' : self.artist,
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