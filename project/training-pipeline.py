import os
import hopsworks
import modal
import numpy
import seaborn
import pandas
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

from tensorflow import keras
from keras.callbacks import EarlyStopping

# If false, run this script on Modal. Otherwise, run locally.
LOCAL=True

FEATURE_GROUP_NAME = "earthquake_pred"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "earthquake_pred_view"
FEATURE_VIEW_DESCRIPTION = "Earthquake prediction dataset"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "earthquake_model"
MODEL_DESCRIPTION = "Earthquake magnitude and depth predictor"
MODEL_VERSION = 1
MODEL_DIRECTORY = MODEL_NAME

loss = 'mse'
optimizer = 'adam'
metrics = ['accuracy']

def plot_history(model_fit_log):
	met = numpy.asarray(model_fit_log.history[metrics[0]])
	val_met = numpy.asarray(model_fit_log.history['val_' + metrics[0]])

	values = numpy.zeros((len(met), 2), dtype=float)
	values[:, 0] = met
	values[:, 1] = val_met

	seaborn.set(style="whitegrid")
	return seaborn.lineplot(data=pandas.DataFrame(
		values,
		pandas.RangeIndex(start=0, stop=len(met)),
		columns=[
			"training-" + metrics[0],
			"validation-" + metrics[0]
		]
	))

def main():
	hw = hopsworks.login()
	feature_store = hw.get_feature_store()

	try:
		feature_view = feature_store.get_feature_view(
			name=FEATURE_VIEW_NAME,
			version=FEATURE_VIEW_VERSION
		)
	except:
		feature_view = feature_store.create_feature_view(
			name=FEATURE_VIEW_NAME,
			description=FEATURE_VIEW_DESCRIPTION,
			version=FEATURE_VIEW_VERSION,
			labels=["depth", "mag"],
			query=feature_store.get_feature_group(
				name=FEATURE_GROUP_NAME,
				version=FEATURE_GROUP_VERSION
			).select_all()
		)

	X_train, X_test, y_train, y_test = feature_view.train_test_split(0.15)

	model = keras.models.Sequential([
		keras.layers.Dense(32, activation="elu", kernel_initializer="he_normal", input_shape=(X_train.shape[1],)),
		keras.layers.Dropout(rate=0.2),
		keras.layers.Dense(32, activation="elu", kernel_initializer="he_normal"),
		keras.layers.Dropout(rate=0.2),
		keras.layers.Dense(32, activation="elu", kernel_initializer="he_normal"),
		keras.layers.BatchNormalization(),
		keras.layers.Dropout(rate=0.2),
		keras.layers.Dense(y_train.shape[1], activation='softmax')
	])

	train_cb = EarlyStopping(monitor=metrics[0], min_delta=0.01, patience=3)

	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
	log = model.fit(X_train, y_train, batch_size=50, validation_split=0.15, epochs=30, callbacks=train_cb)
	report = model.evaluate(X_test, y_test)

	y_pred = model.predict(X_test)
	print(y_pred)
	print(y_test)
	exit()

	#plot_history(log)

	if not os.path.isdir(MODEL_DIRECTORY):
		os.mkdir(MODEL_DIRECTORY)
	model.save(os.path.join(MODEL_DIRECTORY, MODEL_NAME + ".h5"))

	schema_X = Schema(X_train)
	schema_y = Schema(y_train)
	model_schema = ModelSchema(schema_X, schema_y)

	registry = hw.get_model_registry()
	hw_model = registry.python.create_model(
		name=MODEL_NAME,
		description=MODEL_DESCRIPTION,
		version=MODEL_VERSION,
		metrics={metrics[0]: report[1]},
		model_schema=model_schema
	)

	hw_model.save(MODEL_DIRECTORY)

if __name__ == "__main__":
	if LOCAL:
		main()
	else:
		stub = modal.Stub()
		image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "tensorflow"])

		@stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
		def modal_main():
			main()

		with stub.run():
			modal_main()