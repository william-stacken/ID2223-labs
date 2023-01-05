import requests
import csv
import pandas
import hopsworks
import modal

from datetime import datetime
from datetime import timedelta

# If false, run this script on Modal. Otherwise, run locally.
LOCAL=True

# If true, backfill data. Otherwise, load new data from the API and upload it to Hopsworks
BACKFILL=True

# If BACKFILL is True, backfill data either from the API to a local CSV file or from a local CSV file to Hopsworks
TO_CSV=False

BACKFILL_CSV_FILE = \
"https://raw.githubusercontent.com/william-stacken/ID2223-labs/project/project/backfill.csv"

FEATURE_GROUP_NAME = "earthquake_pred"
FEATURE_GROUP_DESCRIPTION="Earthquake prediction dataset"
FEATURE_GROUP_VERSION=1

def api_request_df():
	columns = ["latitude", "longitude", "depth", "mag", "time"]

	if BACKFILL:
		starttime = "2022-11-06"
		endtime = "2022-12-28"
	else:
		now = datetime.now()
		starttime = (now - timedelta(days=1)).strftime("%Y-%m-%d")
		endtime = now.strftime("%Y-%m-%d")

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

	df = pandas.DataFrame.from_records(ourdata, columns=columns)

	return df

def dataframe_cleaning(df):
	# Drop incomplete entries and normalize the timestamp
	# to prevent gradient explosion
	df = df.dropna()
	df["time"] = df["time"] / 1E10

	return df

def main():
	if BACKFILL and not TO_CSV:
		df = pandas.read_csv(BACKFILL_CSV_FILE)
	else:
		df = api_request_df()

	if BACKFILL and TO_CSV:
		df.to_csv(csv_file)
	else:
		df = dataframe_cleaning(df)

		hw = hopsworks.login()
		feature_store = hw.get_feature_store()
		feature_group = feature_store.get_or_create_feature_group(
			name=FEATURE_GROUP_NAME,
			description=FEATURE_GROUP_DESCRIPTION,
			version=FEATURE_GROUP_VERSION,
			primary_key=["latitude", "longitude", "time"]
		)
		feature_group.insert(df, write_options={"wait_for_job": False})

if __name__ == "__main__":
	if LOCAL:
		main()
	else:
		stub = modal.Stub()
		image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4"])

		@stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
		def modal_main():
			main()

		with stub.run():
			modal_main()
