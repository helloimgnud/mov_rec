import pandas as pd

def load_movielens_1m():
    RATINGS_PATH = 'ml-1m/ratings.dat'
    USERS_PATH = 'ml-1m/users.dat'
    ITEMS_PATH = 'ml-1m/movies.dat'
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(RATINGS_PATH, sep='::', names=ratings_cols, engine='python', encoding='latin-1')

    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(USERS_PATH, sep='::', names=users_cols, engine='python', encoding='latin-1')

    items_cols = ['item_id', 'title', 'genres']
    items = pd.read_csv(ITEMS_PATH, sep='::', names=items_cols, engine='python', encoding='latin-1')

    items['genres'] = items['genres'].str.split('|')
    
    return ratings, users, items