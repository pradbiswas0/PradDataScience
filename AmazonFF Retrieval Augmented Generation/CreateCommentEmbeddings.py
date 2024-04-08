import os
import dotenv
import chromadb
import json
from tqdm.auto import tqdm
import pandas as pd
import random
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# load dataset from data/ folder to pandas dataframe
# dataset contains column names

df_amazon = pd.read_csv('reviews.csv')

# remove columns which are not needed. We only need Id, Summary and Text fields
df_comments=df_amazon.drop(['ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'],axis=1)

# If required (e.g. to do a demo), use the instruction below to process a subset of records
# df_comments_reduced=df_comments[:10000]

# convert dataframe to list of dicts with Id and Text columns only
amazon_comments_dict = df_comments.to_dict(orient="records")

# We will be using SentenceTransformer (all-mpnet-base-v2) for generating embeddings that we will store to a chroma document store.

embedding_model1 = SentenceTransformer('all-mpnet-base-v2')
class MyEmbeddingFunction1(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        batch_embeddings1 = embedding_model1.encode(input)
        return batch_embeddings1.tolist()

embed_fn = MyEmbeddingFunction1()

# Initialize the chromadb directory, and client.
client = chromadb.PersistentClient(path="./chromadb")

# create collection
collection = client.get_or_create_collection(
    name=f"amazon_ff_comments2",
    embedding_function=embed_fn
)

# Generate embeddings in batches:

# Generate embeddings, and index titles in batches
batch_size = 50

# loop through batches and generated + store embeddings
for i in tqdm(range(0, len(amazon_comments_dict), batch_size)):

    i_end = min(i + batch_size, len(amazon_comments_dict))
    batch = amazon_comments_dict[i : i + batch_size]

    # Replace title with "No Title" if empty string
    #batch_titles = [str(comment["Summary"]) if str(comment["Summary"]) != "" else "No Title" for comment in batch]
    batch_titles = [str(comment["Summary"]+"."+comment["Text"]) for comment in batch]
    batch_ids = [str(comment["Id"]) for comment in batch]

    # generate embeddings
    batch_embeddings1 = embedding_model1.encode(batch_titles)

    # upsert to chromadb
    collection.upsert(
        ids=batch_ids,
        documents=batch_titles,
        embeddings=batch_embeddings1.tolist(),
    )

# At the end of this script we have a ChromaDB collection called amazon_ff_comments2 whioch stores the embeddings for all the Amazon Fast Foods customer comments. This can now be accessed by other modules as part of RAG.