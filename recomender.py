import pandas 
from pandas import DataFrame as df
import numpy as np 

class Popularity_reccomendation:
    def __init__(self):
        self._user_artist_dataset = None
        self._user_id = None
        self._song_id = None
        self._popularity_reccomend = None
    

    def reccomend_by_popularity(self,user_artist_dataset:df, user_id, song_id):
        self._user_artist_dataset = user_artist_dataset
        self._user_id = user_id
        self._song_id = song_id

        #Get the listen count for song the user listen to and saved the 

        listen_count = self._user_artist_dataset.groupby([self._song_id]).agg({self._user_id: 'count'}).reset_index()
        
        #Change the names of the listen_count dataframe
        listen_count.rename(columns={'user_id':'user_listen_count'}, inplace=True)
        
        #sort the listen in ascending order 
        sort_listen_count=listen_count.sort_values(by=['user_listen_count',self._song_id],ascending=[0,1])

        #reccomend the first ten songs
        
        #sort_listen_count['Rank']= sort_listen_count['user_listen_count'].rank(ascending=0,method='first')
        self._popularity_reccomend = sort_listen_count.head(10) 
        return sort_listen_count


class item_base_reccomendation:

    def __init__(self):
        self._user_artist_dataset = None
        self._user_id = None
        self._song_id = None
        self.coo_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    def load_dataset(self,user_artist_dataset:df, user_id, song_id):
        self._user_artist_dataset = user_artist_dataset
        self._user_id = user_id
        self._song_id = song_id

    #Get all the song a user listens to
    def get_user_playlist(self, user):
        user_data = self._user_artist_dataset[self._user_artist_dataset[self._user_id] == user]
        user_items = list(user_data[self._song_id].unique())
        
        return user_items
        
    #Get all the listener of this song 
    def get_song_listeners(self, song):
        item_data = self._user_artist_dataset[self._user_artist_dataset[self._song_id]== song]
        item_users = set(item_data[self._user_id])
            
        return item_users
        
    #Get all songs
    def get_all_songs(self):
        all_items = list(self._user_artist_dataset[self._song_id].unique())
            
        return all_items

    def construct_cooccurence_matrix(self, user_playlist, all_songs):

        user_songs_listeners=[]
        #Get all the listener of this this user's song and save it into a list
        for i in range(len(user_playlist)):
            user_songs_listeners.append(self.get_song_listeners(user_playlist[i]))
        
        #create a zero matrix of size (songs the user listen) x (all songs in the dataset)
            
        coo_matrix = np.matrix(np.zeros(shape=(len(user_playlist), len(all_songs))), float)

        for i in range(len(all_songs)):
            song_i_listeners =self.get_song_listeners(all_songs[i])

            for j in range(len(user_playlist)):
                user_song_listeners = user_songs_listeners[j]
                #number of users who listen to the two songs
                intersection = song_i_listeners.intersection(user_song_listeners)

                if len(intersection) !=0: #calculating the jaccard coefficient
                    union = song_i_listeners.union(user_song_listeners)
                    coo_matrix[j,i] = float(len(intersection))/float(len(union))
                else:
                    coo_matrix[j,i] = 0

        return coo_matrix

    def top_recommendations(self, user, coo_matrix, all_songs, user_playlist):
        print("Non zero values in coo_matrix :%d" % np.count_nonzero(coo_matrix))

        #average of jaccard coefficient of every song in the dataset

        avg_jaccard = coo_matrix.sum(axis=0)/float(coo_matrix.shape[0])
        avg_jaccard = np.array(avg_jaccard)[0].tolist()

        #sorts all elements in descending order
        sort_index = sorted(((e,i) for i,e in enumerate(list(avg_jaccard))), reverse=True)

        columns = ['user_id', 'song', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_playlist and rank <= 10:

                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def recommend(self, user):
        
        user_playlist = self.get_user_playlist(user)    
            
        print("No. of unique songs from the user playlist: %d" % len(user_playlist))
        
        all_songs = self.get_all_songs()
        
        print("no. of unique songs in the data base: %d" % len(all_songs))
         
        
        coo_matrix = self.construct_cooccurence_matrix(user_playlist, all_songs)
        
        
        df_recommendations = self.top_recommendations(user, coo_matrix, all_songs, user_playlist)
                
        return df_recommendations
    

    def get_similar_items(self, song_list):
        
        user_playlist = song_list
        
        all_songs = self.get_all_songs()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        coo_matrix = self.construct_cooccurence_matrix(user_playlist, all_songs)
        df_recommendations = self.top_recommendations("", coo_matrix, all_songs, user_playlist)
         
        return df_recommendations


        
