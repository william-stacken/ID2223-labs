import os
import modal
import hopsworks
import pandas
import dataframe_image
import seaborn
import numpy
from matplotlib import pyplot

from tensorflow import keras

from datetime import datetime
from datetime import timedelta

# If false, run this script on Modal. Otherwise, run locally.
LOCAL=True

FEATURE_GROUP_NAME = "earthquake_pred"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "earthquake_pred_view"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "earthquake_model"
MODEL_VERSION = 1

PRED_FEATURE_GROUP_NAME = "earthquake_pred_monitoring"
PRED_FEATURE_GROUP_VERSION = 1
PRED_FEATURE_GROUP_DESCRIPTION = "Earthquake magnitude and depth predictor prediction monitoring"

PRED_TABLE_FILE = "eq_last_pred.png"
PRED_TABLE_LENGTH = 50

HEATMAP_FILE = "heatmap.png"
HEATMAP_LATLONG_STEP = 10

if not LOCAL:
	stub = modal.Stub()
	image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "dataframe-image", "tensorflow", "seaborn"])

	@stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
	def modal_main():
		main()

def heatmap_create(model, time):
	t = time.timestamp() / 1E7

	lats = [i for i in reversed(range(-90, 90 + 1, HEATMAP_LATLONG_STEP))]
	longs = [i for i in range(-180, 180 + 1, HEATMAP_LATLONG_STEP)]

	x = []
	for lat in lats:
		for lon in longs:
			x.append([lat, lon, t])

	mags = model.predict(numpy.asarray(x), verbose=0)

	mag_map = numpy.asarray([m[0] for m in mags]).reshape(len(lats), len(longs))

	df = pandas.DataFrame(mag_map, index=lats, columns=longs)
	pyplot.figure(figsize = (10, 7))
	pyplot.title('Magnitude Heatmap')
	ax = seaborn.heatmap(df, linewidth=0.5)
	pyplot.xlabel('longitude')
	pyplot.ylabel('latitude')
	pyplot.savefig(HEATMAP_FILE)


def main():
	hw = hopsworks.login()
	registry = hw.get_model_registry()

	model_dir = registry.get_model(MODEL_NAME, version=MODEL_VERSION).download()
	model = keras.models.load_model(os.path.join(model_dir, MODEL_NAME + ".h5"))

	feature_store = hw.get_feature_store()
	feature_view = feature_store.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
	batch_data = feature_view.get_batch_data()

	feature_group = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
	last_df = feature_group.read()
	last_df = last_df.tail(PRED_TABLE_LENGTH)

	y_pred = model.predict(batch_data)
	y_pred = y_pred[-PRED_TABLE_LENGTH:]

	pred_feature_group = feature_store.get_or_create_feature_group(
		name=PRED_FEATURE_GROUP_NAME,
		version=PRED_FEATURE_GROUP_VERSION,
		description=PRED_FEATURE_GROUP_DESCRIPTION,
		primary_key=["datetime", "predicted_mag", "actual_mag"],
	)

	now = datetime.now()
	pred_df = pandas.DataFrame({
		'datetime': [now.strftime("%Y-%m-%d %H:%M:%S")] * PRED_TABLE_LENGTH,
		'predicted_mag': [pred[0] for pred in y_pred],
		'actual_mag': [act for act in last_df["mag"]]
	})

	pred_feature_group.insert(pred_df, write_options={"wait_for_job" : False})

	dataframe_image.export(pred_df, PRED_TABLE_FILE, table_conversion = 'matplotlib')
	heatmap_create(model, now)

	hw_dataset_api = hw.get_dataset_api()
	hw_dataset_api.upload(PRED_TABLE_FILE, "Resources/images", overwrite=True)
	hw_dataset_api.upload(HEATMAP_FILE, "Resources/images", overwrite=True)

if __name__ == "__main__":
	if LOCAL:
		main()
	else:
		with stub.run():
			modal_main()