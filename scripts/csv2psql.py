import psycopg2
import csv
import sys
import datetime

csv_data = csv.DictReader(open(sys.argv[1],'r'))
next(csv_data,None) # Skip the headers

database = psycopg2.connect (database = "machinelearning", user="gisuser", password="something", host="localhost", port="5432")
cursor = database.cursor()

for row in csv_data:
    registered = 0
    if row['Member Type'] == 'Registered':
        registered = 1
    entry = [ datetime.datetime.strptime(row['Start date'], '%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S'), registered]
    cursor.execute("INSERT INTO cabi_trips (embark_date, registered) VALUES (%s,%s)", entry)

cursor.close()
database.commit()
database.close()

print("CSV data imported")
