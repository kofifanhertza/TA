import mysql.connector 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="kh670205"
)

mycursor = mydb.cursor()


# mycursor.execute("CREATE DATABASE IF NOT EXISTS testdb")
#mycursor.execute("USE testdb")

# mycursor.execute("""
#     CREATE TABLE IF NOT EXISTS detectionData (
#         detection_id INTEGER,
#         room_id INTEGER,
#         timestamp VARCHAR(255),
#         n_in INTEGER,
#         n_out INTEGER,
#         n_current INTEGER
#     )
# """)
mycursor.execute("DROP DATABASE testdb")
#mycursor.execute("DROP TABLES detectionData")
# sql = "INSERT INTO detectionData (detection_id, room_id, timestamp, n_in, n_out, n_current) VALUES (%s, %s, %s, %s, %s, %s)"
# val = (1, 101, '2024-06-09 12:00:00', 5, 2, 3)

# # Execute the query
# mycursor.execute(sql, val)

# Commit the changes to the database
mydb.commit()
