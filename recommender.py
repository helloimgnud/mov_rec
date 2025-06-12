import logging
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.load_data import load_movielens_1m
import json
from model_in_use import AutoRec, UserFeaturesNet
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MyDataset(Dataset):
    def __init__(self, user_item_mat, user_indices):
        self.user_item_mat = user_item_mat
        self.user_indices = user_indices

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_id = self.user_indices[idx]
        return self.user_item_mat[user_id]

def get_predicted_ratings(model, user_item_mat, user_indices, device):
    model.eval()
    preds = []
    dataset = MyDataset(user_item_mat, user_indices)
    batch_size = 256
    num_workers = 2
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for r in dl:
            r = r.to(device)
            pred = model(r)
            preds.append(pred.cpu())
    return torch.cat(preds, dim=0)

def get_recommendations(age, gender, occupation):
    
    try:
        print('data input to recommender: ')
        print(gender)
        print(age)
        print(occupation)

        try:
            autorec = torch.load('autorec_full_v2.pth', weights_only = False)
            print("Autorec model loaded successfully.")

            data = torch.load('dataset_full.pt')
            print("Dataset loaded successfully.")

        except Exception as e:
            print("Error loading model or data:", e)

        try:
            existed_users = data['train_users']
            user_item_mat = data['user_item_mat']
            existed_predicted_ratings = get_predicted_ratings(autorec, user_item_mat, existed_users, device)

            print("existed_predicted_ratings done")
        except Exception as e:
            print("Error existed_predicted_ratings :", e)

        try:
            features_net = torch.load('features_net_full_v2.pth', weights_only = False)
            print("Features net loaded successfully.")
            features_net.eval()
            input_tensor = torch.tensor([[age, gender, occupation]], dtype=torch.float32)
            weights = features_net(input_tensor)
            preds = torch.matmul(weights, existed_predicted_ratings)

        except Exception as e:
            print("Error in features net:", e)
        
        print("weights:", weights)
        print("preds:", preds)

        try:
            try:
                ___, __, items_df = load_movielens_1m()
                topk_values, topk_indices = preds[0].topk(30)
                
                topk_item_id = [int(k) for k in topk_indices[10:30]]

            except Exception as e:
                print("Error in calculate top K:", e)


            print("topk_item_id:", topk_item_id)

            items = pd.read_csv("ml-1m/items.csv")
            print("items columns:", items.columns)
            print("items sample:", items.head())


            matched_items = items[items['item_id'].isin(topk_item_id)]
            matched_items_ordered = matched_items.set_index('item_id').loc[topk_item_id].reset_index()
            print("matched_items shape:", matched_items.shape)
            print(matched_items[['item_id', 'tmdbId']] if not matched_items.empty else "No matched items found")

            # items = pd.read_csv("ml-1m/items.csv")
            # matched_items = items[items['item_id'].isin(topk_item_id)]
            topk_tmdb_ids = matched_items_ordered['tmdbId'].tolist()
            topk_tmdb = [int(k) for k in topk_tmdb_ids]

            print(topk_tmdb)

            print("topk_item_id:", topk_item_id)

        except Exception as e:
                print("Error in tmdb_id:", e)

        

        return topk_tmdb;

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)


