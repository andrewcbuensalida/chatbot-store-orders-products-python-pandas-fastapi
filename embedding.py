import pandas as pd
import numpy as np
import os
from typing import List
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
# To create the index. Don't run this cell if you want to use the existing index.
from pinecone import ServerlessSpec, Pinecone

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "chatbot-store"  # put in the name of your pinecone index here. When creating the index in pinecone.io, the Dimensions have to be the same as the result length.

# connect to index
index = pc.Index(index_name)

client = OpenAI(max_retries=5, api_key=os.environ.get("OPENAI_API_KEY"))
embedding_model = "text-embedding-3-small"

def search_pinecone(query,top_k=10):
    query_embedding = client.embeddings.create(input=query, model=embedding_model).data[0].embedding
    results = index.query(vector=[query_embedding], top_k=top_k, include_metadata=True)
    # Convert the matches to a DataFrame
    matches_df = pd.DataFrame(
        [match["metadata"] for match in results["matches"]]
    )
    matches_df = matches_df.drop(columns=["combined"])

    return matches_df


# Deprecated. Use search_pinecone instead.
def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Deprecated. search for a specific product
def search_embeddings(df: pd.DataFrame, query, n=10, pprint=True):
    embedding = get_embedding(query, model="text-embedding-3-small")
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values("similarities", ascending=False).head(n)
    res = res.drop(columns=["combined", "embedding"])
    return res
