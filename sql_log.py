import mysql.connector
import time
import os

def run_query():
    # Connect to the database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="kh670205",
        database="testdb"  # Ensure you are connected to the correct database
    )

    mycursor = mydb.cursor()

    # Run the SELECT query
    mycursor.execute("SELECT * FROM detectionData")

    # Fetch all results
    results = mycursor.fetchall()

    # Print the results
    print("DETECTION ID, ROOM_ID, TIMESTAMP, IN_COUNTER, OUT_COUNTER, CURRENT_COUNTER")
    for row in results:
        print(row)

    # Close the cursor and connection
    mycursor.close()
    mydb.close()

while True:
    os.system('cls')
    run_query()
    time.sleep(1)  # Wait for 10 seconds before running the query again