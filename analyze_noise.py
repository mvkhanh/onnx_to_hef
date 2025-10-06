import os

import numpy as np

from hailo_sdk_client import ClientRunner

data_path = "calib_set/1024_afad.npy"
data = np.load(data_path)
assert os.path.isfile(data_path), "Please provide valid path for a dataset"
runner = ClientRunner(har='agegender/agegender_compiled_model.har')

runner.analyze_noise(data_path, batch_size=2, data_count=64)  # Batch size is 1 by default
model_name = 'agegender'
har_path = model_name + "_noise_compile.har"
runner.save_har(har_path)