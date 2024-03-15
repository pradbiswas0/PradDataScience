# Create embeddings from summary and text (detailed comments) fields from Amazon Fine Foods Product Review dataset

# Encode contents of the Text column and store in a csv file
import pandas as pd # dataframe manipulation
import numpy as np # linear algebra
from sentence_transformers import SentenceTransformer
df_amazon_raw = pd.read_csv('Reviews.csv')
df_reviews=df_amazon_raw[['Text']]

def compile_text(x):
    text =  f"""Text: {x['Text']}"""
    return text

sentences = df_reviews[['Text']].apply(lambda x: compile_text(x), axis=1).tolist()
model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")

output = model.encode(sentences=sentences, show_progress_bar= True, normalize_embeddings  = True)
df_embedding = pd.DataFrame(output)
df_embedding.to_csv("embedding_train_amazon.csv",index = False)

# Encode contents of the Summary column and store in a csv file
def compile_text_summ(x):
    text =  f"""Summary: {x['Summary']}"""
    return text

df_reviews_summ=df_amazon_raw[['Summary']]

sentences_summ = df_reviews_summ[['Summary']].apply(lambda x: compile_text_summ(x), axis=1).tolist()

output_summ = model.encode(sentences=sentences_summ, show_progress_bar= True, normalize_embeddings  = True)

df_embedding_summ = pd.DataFrame(output_summ)
df_embedding_summ.to_csv("embedding_train_amazon_summ.csv",index = False)