import pandas as pd
import numpy as np 

class reccomendation:
    def __init__(self):
        self._user_artist_dataset = None
        self._user_id = None
        self._song_id = None
        self.all_songs = None 
    def load_dataset(self,user_artist_dataset, user_id, song_id):
        self._user_artist_dataset = user_artist_dataset
        self._user_id = user_id
        self._song_id = song_id
        self.all_songs = list(self._user_artist_dataset[self._song_id].unique())


    def reccomend_by_popularity(self):

        #Get the listen count for song the user listen to and saved the 
        listen_count = self._user_artist_dataset.groupby([self._song_id]).agg({self._user_id: 'count'}).reset_index() 
      
        listen_count.rename(columns={'user_id':'user_listen_count'}, inplace=True) #Change the names of the listen_count dataframe
          
        sort_listen_count=listen_count.sort_values(by=['user_listen_count',self._song_id],ascending=[0,1])  #sort the listen in ascending order 
 
        return sort_listen_count

    """
        Item base reccomendaiton below
    """

    #Get all the song a user listens to
    def get_user_playlist(self, user):
        user_data = self._user_artist_dataset[self._user_artist_dataset[self._user_id] == user]
        user_items = list(user_data[self._song_id].unique())
        
        return user_items
        
    #Get all the listeners of this song 
    def get_song_listeners(self, song):
        item_data = self._user_artist_dataset[self._user_artist_dataset[self._song_id]== song]
        item_users = set(item_data[self._user_id])
            
        return item_users
        
    #Get all songs
    def get_all_songs(self):
        all_items = list(self._user_artist_dataset[self._song_id].unique())
            
        return all_items

    def construct_cooccurence_matrix(self, user_playlist):

        user_songs_listeners=[]
        #Get all the listener of this this user's song and save it into a list
        for i in range(len(user_playlist)):
            user_songs_listeners.append(self.get_song_listeners(user_playlist[i]))
        
        #create a matrix (songs the user listen) x (all songs in the dataset)     
        coo_matrix = np.matrix(np.zeros(shape=(len(user_playlist), len(self.all_songs))), float)

        for i in range(len(self.all_songs)):
            song_i_listeners =self.get_song_listeners(self.all_songs[i])

            for j in range(len(user_playlist)):
                song_j_listener = user_songs_listeners[j]        
                intersection = song_i_listeners.intersection(song_j_listener)#number of users who listen to the two songs

                if len(intersection) !=0: 
                    union = song_i_listeners.union(song_j_listener) #people who listen to song i and the user's song
                    coo_matrix[j,i] = float(len(intersection))/float(len(union)) 
                else:
                    coo_matrix[j,i] = 0

        return coo_matrix

    def top_recommendations(self, user, coo_matrix, user_playlist):
        if(np.count_nonzero(coo_matrix)==0):
            error_message = ["No match found"]
            message = pd.DataFrame(error_message)
            return message
        print("\nNon zero values in coo_matrix                : %d\n" % np.count_nonzero(coo_matrix))

        #sum up and average all the jaccard values of each column into a list
        avg_jaccard = coo_matrix.sum(axis=0)/float(coo_matrix.shape[0]) 
        avg_jaccard = np.array(avg_jaccard)[0].tolist()

        #sorts all elements in descending order
        sort_index = sorted(((e,i) for i,e in enumerate(avg_jaccard)), reverse=True)
        df = pd.DataFrame(columns=[ 'song', 'score', 'rank'])#,'user_id'])
        
        rank = 1 
        for i in range(len(sort_index)):
            if not np.isnan(sort_index[i][0]) and self.all_songs[sort_index[i][1]] not in user_playlist and rank <= 10:
                df.loc[len(df)]=[self.all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank +=1
                
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def recommend_by_playlist(self, user):
        
        user_playlist = self.get_user_playlist(user)   

        print("\nNo. of unique songs in user's playlist       : %d" % len(user_playlist)) 
        print("\nno. of unique songs in the data base         : %d" % len(self.all_songs))
              
        coo_matrix = self.construct_cooccurence_matrix(user_playlist)
        df_recommendations = self.top_recommendations(user, coo_matrix, user_playlist)
      
        return df_recommendations
    

    def get_similar_songs(self, song_list:list):
        
        print("\nno. of unique songs in the data base         : %d" % len(self.all_songs))
        coo_matrix = self.construct_cooccurence_matrix(song_list)
        df_recommendations = self.top_recommendations("", coo_matrix, song_list)
         
        return df_recommendations


        
