import sys 
sys.path.append("preprocessing")
sys.path.append("dataset")

from nlp_tools import NLP_Preprocessing
from steamdataset import SteamDataset

import numpy as np

import gensim.downloader
import psycopg2

class Glove25(SteamDataset):

    def __init__(self, path=None, n_rows=None, seed=None, lemm=True):

        if not path is None:
            super().__init__(path, n_rows, seed, lemm)

        self.model = gensim.downloader.load("glove-twitter-25")
            
    def get_embedding(self, n_documents=30):
        """
        Lấy token trực tiếp 
        """
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
    
    def get_query_embedding(self, query: list[str]|str)->np.ndarray:
        # Chuyển query sang dạng token
        if isinstance(query, str):
            query = [query] # Cho giống với dạng trong NLP Procesing
            tools = NLP_Preprocessing()
            query_tokens = tools.preprocess(query, True)
            query_tokens = query_tokens[0] # Kết quả trả về là list[str[str]]

        valid_tokens = [word for word in query_tokens if word in self.model]
        
        if len(valid_tokens) == 0:
            return None
        else:
            word_emmbedding = [self.model[word] for word in valid_tokens]
        
        word_emmbedding = np.array(word_emmbedding)
        results = np.sum(word_emmbedding, axis=0)/word_emmbedding.shape[0]

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
        """
        Tạo bảng trong psql
        """
        s = """
            CREATE TABLE IF NOT EXISTS glove_25  (
                id bigserial PRIMARY KEY, 
                embedding vector(25), 
                document TEXT
            );
        """

        with self.conn.cursor() as cur:
            cur.execute(s)
        
        self.conn.commit()

    def store_embeddings(self, embeddings: np.ndarray, documents: list[str]|np.ndarray):
        """
        Lưu trữ embedding và documents vào psql
        """
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

    def search_query(self, model: Glove25, query: str=None):
        """
        Tìm các kết quả gần nhất bằng query, thực hiện embedding câu như trên
        """
        sent_embeddings = model.get_query_embedding(query=query)
        sent_embeddings = sent_embeddings.tolist()

        # Trong trường hợp không thể embedding thì trả về None
        if sent_embeddings is None:
            return None
        
        sql = """
            SELECT * 
            FROM glove_25
            ORDER BY embedding <-> %s::vector
            LIMIT 5;
        """
        values = (sent_embeddings, )
        with self.conn.cursor() as cur:
            cur.execute(sql, values)
            results = cur.fetchall()
            
        return results
        

if __name__ == "__main__":
    glove25 = Glove25("../dataset.csv", 100_000, 42, True)

    sent_embeddings = glove25.get_embedding(n_documents=100_000)
    documents = glove25.get_documents(100_000)

    db = PGVector(
        dbname="steamdataset",
        user="postgres",
        password="postgres",
        host="localhost",  
        port="5432"
    )
    db.create_table()
    db.store_embeddings(sent_embeddings, documents)

    results = db.search_query(glove25, 'Cats, bears, dogs, monkeys')
    if results is None:
        print("Xin lỗi, toàn bộ từ khoá không có trong từ điển")
    for _, _, document in results:
        print(document, end="\n\n")
    