from pysr import PySRRegressor
import numpy as np

feature_1 = np.load("../salt_features.npy") # 14
feature_2 = np.load("../solvent_features.npy") # 14
feature_3 = np.load("../condition_features.npy") # 6


important_1 = feature_1[:, [1, 5]] # logP and C_NHOH of Li-salt
important_2 = feature_2[:, [3, 8]] # RB and HAM of solvent
important_3 = feature_3[:,[0, 5]] # T and c

X_selected = np.hstack((important_1, important_2, important_3)) # input feature vector

model = PySRRegressor.from_file(run_directory="/home/zlwang/project/salt/model/A-model/symbolic/outputs/2")

i = 29 # 30th symbolic equation
prediction = model.predict(X_selected, index=i)
