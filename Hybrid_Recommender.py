

import pandas as pd
pd.set_option('display.max_columns', 20)

movie = pd.read_csv('Bootcamp/4.Hafta/Ders Öncesi Notlar/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Bootcamp/4.Hafta/Ders Öncesi Notlar/movie_lens_dataset/rating.csv')

rating.head()
movie.head()
df = movie.merge(rating, how="left", on="movieId")
df.head()

# User Movie Df

df.shape

df["title"].nunique()

df["title"].value_counts().head()

rating_counts = pd.DataFrame(df["title"].value_counts())
rating_counts.head()
rare_movies = rating_counts[rating_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
common_movies.head()
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape

user_movie_df.head(10)

user_movie_df.columns

len(user_movie_df.columns)


#######
# Item-Based movie suggestion
#######

movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# random movie selection
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

#######
# Finalized item based recommendation with scripting
#######

def create_user_movie_df():
    movie = pd.read_csv('Bootcamp/4.Hafta/Ders Öncesi Notlar/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Bootcamp/4.Hafta/Ders Öncesi Notlar/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df_functioned = create_user_movie_df()

def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

movies_from_item_based = item_based_recommender(movie_name, user_movie_df_functioned)

# how many film should we recommend
rec_count = 5

movies_from_item_based[1:(rec_count+1)].index

# Again scripting

def suggession_list (movie_name,user_movie_df,is_random = True,rec_count = 5):
    if is_random:
        movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

    movies_from_item_based = item_based_recommender(movie_name, user_movie_df)
    print(f"if you like {movie_name}, you may like those:")
    return movies_from_item_based[1:(rec_count + 1)].index

suggession_list("Fierce Creatures (1997)",user_movie_df_functioned,is_random=False,rec_count=5)
suggession_list("any",user_movie_df_functioned,rec_count=5)

##############
# GÖREV 2.2 - User-Based movie suggestion - Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
##############

random_user = int(pd.Series(user_movie_df_functioned.index).sample(1, random_state=45).values)

random_user_df = user_movie_df_functioned[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

##############
# GÖREV 3.2 - User-Based movie suggestion - Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
##############

movies_watched_df = user_movie_df_functioned[movies_watched]
movies_watched_df.head()

movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count.head()

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

users_same_movies.head()
users_same_movies.count()
users_same_movies.index

##############
# GÖREV 4.2 - User-Based movie suggestion - Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
##############

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])
final_df.head()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df.head()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.60)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head(50)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

##############
# GÖREV 5.2 - User-Based movie suggestion - Weighted Average Recommendation Score'un Hesaplanması
##############

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df.head()

recommendation_df[["movieId"]].nunique()
recommendation_df.sort_values("weighted_rating", ascending=False).head(15)

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.4].sort_values("weighted_rating", ascending=False)

Top_movies_for_recomendation = movies_to_be_recommend.merge(movie[["movieId", "title"]])

##############
# GÖREV 6 - Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre
#           5 öneri user-based, 5 öneri item-based olacak şekilde 10 öneri yapınız.
##############

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"].values[0]
Random_movie_name = movie[movie["movieId"] == movie_id]["title"].values[0]

Top_5_user_based = Top_movies_for_recomendation["title"].values[0:5].tolist()

Top_5_item_based = suggession_list(Random_movie_name,user_movie_df_functioned,is_random=False,rec_count=5).tolist()





