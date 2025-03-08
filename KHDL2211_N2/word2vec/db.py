import psycopg2

try: 
    conn = psycopg2.connect(
        dbname="word2vec",
        user="postgres",
        password="postgres",
        host="localhost",  # Change if using a remote server
        port="5432"
    )
    
    cur = conn.cursor()
    cur.execute("SELECT * FROM steamcorpus;")
    
    rows = cur.fetchall()  # Fetch all rows
    
    for row in rows:
        print(row)  

except Exception as e:
    print("Error:", e)

finally:
    cur.close()
    conn.close()
