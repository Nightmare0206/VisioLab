import psycopg2

conn = psycopg2.connect(
    host="database-3.cxxtqrexryzl.us-east-1.rds.amazonaws.com",
    database="dbname",
    user="postgres1",
    password="programming",
    port=5432
)