import graphlab as gl
import numpy as np
import pandas as pd
import pickle, json
from collections import defaultdict



def determine_input(input_str, df_data, user_map):
    try:
        id_val = int(input_str)
        if id_val in df_data.User.values:
            return id_val
        else: 
            return None
    except ValueError:
        if input_str in user_map:
            return user_map[input_str]
        else:
            return None

def user_rated_visited(user_id, df_data, df_raw):
    #Find destinations climber has rated climbs from
    climbs_rated = df_data[df_data.User==user_id].Climb
    visited = set(df_raw.loc[climbs_rated].sub_location.values)
  
    #Finder user difficulty rating range from stared climbs
    user_ratings = df_raw.loc[climbs_rated].rating
    user_rating_std = user_ratings.std()
    user_rating_mean = user_ratings.mean()
    rating_max = user_rating_mean+user_rating_std
    
    return visited, rating_max
   
#Find climb locations to recommend
def rec_loc_climb(user_id, visited, rating_max, df_raw,
                  model, verbose=False, n_areas=3, n_climbs=10):
    
    climb_recs = model.recommend(users=[user_id], k=13000)
    loc_climb_recs = defaultdict(list)
    loc_recs = []
    n_recs = 0
    for rec in climb_recs:
        climb = rec['Climb']
        if df_raw.loc[climb].rating < rating_max:
            loc = df_raw.loc[climb].sub_location
            if loc not in (list(visited) + loc_recs):
                loc_climb_recs[loc] += [climb]
                if len(loc_climb_recs[loc]) == n_climbs:
                    loc_recs += [loc]
                    n_recs += 1
                    if n_recs == n_areas:
                        break
    if verbose:
        for loc in loc_recs:
            print loc
            print loc_climb_recs[loc]
    return loc_recs, loc_climb_recs
    
def get_latent_user(user_id, model):
    coefs = model.get('coefficients')
    df_fac_user = pd.DataFrame(np.array(coefs['User']['factors']))
    df_fac_user.set_index(np.array(coefs['User']['User']), inplace=True)
    return df_fac_user.loc[user_id]

if __name__ == "__main__":
    #user name to user id dict
    with open('user_map.p','r') as f:
        user_map = pickle.load(f)

    #star rating and climb data
    df_data = pd.read_csv('star5.csv')
    with open('df_raw_star5.p','r') as f:
        df_raw = pickle.load(f)

    #load in recommendation models
    sim_mod = gl.load_model('sim_mod')
    rfr_mod = gl.load_model('rfm_mod_15')
    rfr_mod_lf = gl.load_model('rfm_mod_features_extracted')
    
    #user input
    input_str = 'LauraColyer'
    user_id = determine_input(input_str, df_data, user_map)
    
    visited, rating_max = user_rated_visited(user_id, df_data, df_raw)

    #Get climb recomendations from item similarity model
    loc_recs_sim, loc_climb_recs_sim = rec_loc_climb(user_id,
                                                  visited,
                                                  rating_max,
                                                  df_raw,
                                                  sim_mod,
                                                  verbose=True)
    
    #Get climb recomendations from item similarity model
    loc_recs_rfr, loc_climb_recs_rfr = rec_loc_climb(user_id,
                                                  visited,
                                                  rating_max,
                                                  df_raw,
                                                  rfr_mod,
                                                  verbose=True)
    print get_latent_user(user_id, rfr_mod_lf)
    