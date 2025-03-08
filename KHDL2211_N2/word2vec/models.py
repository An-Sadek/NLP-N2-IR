import os 
import sys 
sys.path.append("preprocessing")
sys.path.append("dataset")

from nlp_tools import NLP_Preprocessing
from steamdataset import SteamDataset

import numpy as np

import gensim.downloader
import psycopg2

class Glove25(SteamDataset):

    def __init__(self, path, nrows, seed, lemm=True):
        super().__init__(path, nrows, seed, lemm)
        
        self.model = gensim.downloader.load("glove-twitter-25")

    def get_embedding(self, n_documents=30):
        assert n_documents <= self.n_rows

        small_tokens = self.tokens[:n_documents]
        results = np.zeros((len(small_tokens), 25))

        for idx in range(len(small_tokens)):
            valid_tokens = [word for word in small_tokens[idx] if word in self.model]

            if len(valid_tokens) == 0:
                word_emmbedding = [0] * 25
            else:
                word_emmbedding = [self.model[word] for word in valid_tokens]
            
            word_emmbedding = np.array(word_emmbedding)
            results[idx] = np.sum(word_emmbedding, axis=0)/word_emmbedding.shape[0]

        return results
    
    def get_documents(self, n_documents):
        return self.documents[: n_documents]
    

class PGVector:

    def __init__(self, 
            dbname, 
            user, 
            password, 
            host, 
            port
    ):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,  
            port=port
        )
    
    def create_table(self):
        s = """
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE glove_25 IF NOT EXISTS (
                id bigserial PRIMARY KEY, 
                embedding vector(25), 
                document TEXT
            );
        """

    def store_embeddings(self, embeddings: np.ndarray, documents: list[str]|np.ndarray):
        assert embeddings.ndim == 2, "Phải là ma trận 2 chiều"
        assert embeddings.shape[1] == 25, "Vector phải có 25 phần tử"
        assert len(embeddings) == len(documents), "Kích thước của embeđing và documents bằng nhau"

        sql = """
            INSERT INTO glove_25 (embedding, document)
            VALUES (%s, %s)
        """
        embeddings = embeddings.tolist()
        with self.conn.cursor() as cur:
            for idx in range(len(embeddings)):
                records = (embeddings[idx], documents[idx])
                cur.execute(sql, records)
        self.conn.commit()

    def search_query(model: Glove25, query: str=None):
        tools = NLP_Preprocessing()
        query = tools.preprocess(query, True)
        


if __name__ == "__main__":
    glove25 = Glove25("../dataset.csv", 100, 42, True)

    sent_embeddings = glove25.get_embedding(100)
    documents = glove25.get_documents(100)

    db = PGVector(
        dbname="steamdataset",
        user="postgres",
        password="postgres",
        host="localhost",  
        port="5432"
    )
    db.store_embeddings(sent_embeddings, documents)
    