import pickle
import numpy as np

DATA_PATH = "data/raw/BMCLab.pkl"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

trial = data["SUB01"]["SUB01_off_walk_1"]

pose = trial["pose"]

print("=" * 60)
print("POSE TYPE")
print(type(pose))

print("\nPOSE SHAPE")
print(pose.shape)

print("\nFIRST FRAME")
print(pose[0])

print("\nFIRST FRAME LENGTH")
print(len(pose[0]))