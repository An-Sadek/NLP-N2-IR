import sys 
sys.path.append("word2vec")
from models import PGVector, Glove25


glove25 = Glove25()

db = PGVector(
    dbname="steamdataset",
    user="postgres",
    password="postgres",
    host="localhost",  
    port="5432"
)

results = db.search_query(glove25, 'Cats, bears, dogs, monkeys')
if results is None:
    print("Xin lỗi, toàn bộ từ khoá không có trong từ điển")
for _, _, document in results:
    print(document, end="\n\n")