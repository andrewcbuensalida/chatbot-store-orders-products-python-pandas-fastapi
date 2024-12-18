import pandas as pd
import numpy as np
import os
from typing import List
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

load_dotenv()

client = OpenAI(max_retries=5, api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = client.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# search for a specific product
def search_embeddings(df: pd.DataFrame, query, n=10, pprint=True):
    embedding = get_embedding(query, model="text-embedding-3-small")
    df["similarities"] = df.embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values("similarities", ascending=False).head(n)
    res = res.drop(columns=["combined", "embedding"])
    return res
