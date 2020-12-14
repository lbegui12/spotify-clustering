# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:46:53 2020

@author: Louis
"""
import sys

import spotipy    # la librairie pour manipuler l'api spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyPKCE
from spotipy.exceptions import SpotifyException

from track import Track
from playlist import Playlist

from secret import spotify_user_id
from secret import spotify_secret

import pandas as pd


class SpotifyHelper:
    
    def __init__(self):
        self.client_id = spotify_user_id
        self.client_secret = spotify_secret
        self.sOAut = SpotifyOAuth(
                                    client_id=self.client_id,
                                    client_secret=self.client_secret,
                                    redirect_uri="http://localhost/",
                                    scope="user-library-read playlist-modify-public",
                                    show_dialog=False
                                )
        self.sp = spotipy.Spotify(auth_manager=self.sOAut)
        
    def get_user_saved_tracks(self, limit = 10, verbose = False):  
        try:
            p = Playlist()
            p.set_playlist_name("Saved songs")

            offset = 0
            while(offset<limit):            
                results = self.sp.current_user_saved_tracks(limit=1, offset=offset)
                for idx, item in enumerate(results['items']):
                    track = item['track']
                    t = Track(track)
                    
                    result = self.sp.audio_features(track['id'])
                    t.set_features(result[0])
                    if verbose:
                        t.print_track()         # print on the fly 
                    
                    # Add the track to the created playlist
                    p.add_track(t)
                    offset+=1

            return p
            
        except SpotifyException:
            print("Limit reached")
            return None
        
    def create_playlist(self, _playlist):
        
        playlist_id = "-1"
        try:
            user_id = self.sp.me()['id']
            playlist= self.sp.user_playlist_create(user_id, name=_playlist.name)
            playlist_id= str(playlist['id'])
            self.sp.playlist_add_items(playlist_id, [track.id for track in _playlist.tracks])
        except:
            print("Error while adding items to playlist {}".format(_playlist.name))
        return playlist_id
        
    def del_playlist(self, playlist_id):
        user_id = self.sp.me()['id']
        self.sp.user_playlist_unfollow(user_id, playlist_id)
        
    def del_playlist_by_string(self, strings):       # SUPPR LES PLAYLISTS
        user_id = self.sp.me()['id']
        playlists = self.sp.user_playlists(user_id, limit=50, offset=0)
        
        # go through the user playlists
        while playlists:
            for i, playlist in enumerate(playlists['items']):
                print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['id'],  playlist['name']))
                for s in strings:
                    if s in playlist['name']:
                        self.del_playlist(playlist['id'])
            if playlists['next']:
                playlists = self.sp.next(playlists)
            else:
                playlists = None
    
def main(args=None):
    cp = SpotifyHelper()

    #p = cp.get_user_saved_tracks(limit=1066, verbose=True)
    #print("%d tracks in the playlist" % p.count)
    #data = p.playlist_to_df()
    #print(data.head())
    #data.to_csv("datasets\output\mySavedSongs_{}.csv".format(spotify_user_id), index=False)
 
    to_remove = ["MiniBatchKMeans","AffinityPropagation", "Birch" ,"MeanShift", "OPTICS","Agglo","Spectral","DBSCAN"]
    response = cp.del_playlist_by_string(to_remove)

if __name__ == '__main__':
    main(sys.argv[1:])
    