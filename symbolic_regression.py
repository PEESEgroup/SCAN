from pysr import PySRRegressor
import numpy as np

feature_1 = np.load("data/salt_features.npy") 
feature_2 = np.load("data/solvent_features.npy") 
feature_3 = np.load("data/condition_features.npy") 
target = np.loadtxt("data/conductivity_target.txt")


important_1 = feature_1[:, [1, 5]]
important_2 = feature_2[:, [3, 8]]
important_3 = feature_3[:,[0, 5]]

X_selected = np.hstack((important_1, important_2, important_3))  

#corr = np.corrcoef(X_selected.T)
#print(corr)

y = target 

model = PySRRegressor(
        niterations=1000, 
        parsimony=1e-3,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['sqrt', 'log', 'sin', 'cos'],
)


model.fit(X_selected, y)


