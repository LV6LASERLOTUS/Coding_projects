import pandas as pd 
import recomender as recomender

user_dataset = pd.read_csv("user_dataset.csv", sep=',')
artist_dataset = pd.read_csv("artist_dataset.csv", sep=',')
# print(user_dataset.head())
# print(artist_dataset.head())
user_artist_dataset = pd.merge(
                                user_dataset,
                                artist_dataset.drop_duplicates(subset=['song_id']),
                                on='song_id',how='left'
                            )

user_artist_dataset['song'] = user_artist_dataset['title']+' - '+user_artist_dataset['artist_name']
user_artist_dataset = user_artist_dataset.head(10000)
popularity = recomender.Popularity_reccomendation()
top_songs = popularity.reccomend_by_popularity(user_artist_dataset=user_artist_dataset,user_id='user_id',song_id='song')
print(top_songs.head(10))

item_based = recomender.item_base_reccomendation()
item_based.load_dataset(user_artist_dataset=user_artist_dataset,user_id='user_id',song_id='song')

user_items = item_based.get_user_playlist(user_artist_dataset['user_id'][5])

# for user_item in user_items:
#     print(user_item)

songs = item_based.recommend(user_artist_dataset['user_id'][5])
print(songs.head())

songs = item_based.get_similar_items(['Secrets - OneRepublic'])
print(songs.head())
