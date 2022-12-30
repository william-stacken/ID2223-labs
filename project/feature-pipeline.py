import requests
import csv

starttime = "2022-11-06"
endtime = "2022-12-28"

url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime="+starttime+"&endtime="+endtime

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data={})
myjson = response.json()
ourdata = []

# print(response.status_code)
samples_count = len(myjson["features"])
print("Collected: " + str(samples_count) + " samples")

for x in range(len(myjson["features"])):
    properties = list(myjson["features"][x]["properties"].values())
    coordinates = list(myjson["features"][x]["geometry"].values())[1]

    mag = properties[0]
    time = properties[2]

    latitude = coordinates[0]
    longitude = coordinates[1]
    depth = coordinates[2]

    ourdata.append([latitude, longitude, depth, mag, time])

with open("database.csv", "w", encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["latitude", "longitude", "depth", "mag", "time"])
    writer.writerows(ourdata)
