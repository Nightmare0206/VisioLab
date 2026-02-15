import psycopg2

conn = psycopg2.connect(
    host="database-3.cxxtqrexryzl.us-east-1.rds.amazonaws.com",
    database="dbname",
    user="postgres1",
    password="programming",
    port=5432
)

cur = conn.cursor()

# Query all data
cur.execute("SELECT * FROM face_images;")
rows = cur.fetchall()

col_names = [desc[0] for desc in cur.description]
print("Columns:", col_names)

# Print all rows
for row in rows:
    print(row)

cur.execute("SELECT * FROM otps;")
rows2 = cur.fetchall()

# Print column names
col_names = [desc[0] for desc in cur.description]
print("Columns:", col_names)

# Print all rows
for row in rows:
    print(row)
    
for row in rows2:
    print(row)

cur.close()
conn.close()