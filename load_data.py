from datasets import load_dataset
import pandas as pd
import torch

print("Loading Amazon Electronics Dataset...")

# 1. Load Node Features (Items: Text descriptions, Titles, Image URLs)
# "raw_meta_Electronics" contains the item metadata
amazon_items = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Electronics", split="full")
df_items = amazon_items.to_pandas()

# 2. Load Interactions (Edges: User-Item clicks/reviews)
# "raw_review_Electronics" contains the actual user behaviors
amazon_reviews = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", split="full")
df_edges = amazon_reviews.to_pandas()

# Filter for implicit feedback (e.g., 4 or 5-star ratings = "Click/Positive Edge")
positive_edges = df_edges[df_edges['rating'] >= 4.0]

print(f"Loaded {len(df_items)} items (with text/images).")
print(f"Loaded {len(positive_edges)} positive interactions (edges).")