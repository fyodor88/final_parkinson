import pickle
from collections import Counter

DATA_PATH = "data/raw/DNE.pkl"

print("Loading DNE dataset...")

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print("Number of subjects:", len(data))

print("\n========== DATASET STRUCTURE ==========")

# Inspect the first subject and first trial
first_subject = next(iter(data))
first_trial = next(iter(data[first_subject]))

print("First subject:", first_subject)
print("First trial:", first_trial)

trial = data[first_subject][first_trial]

print("\nTrial keys:")
print(trial.keys())

print("\n========== LABELS ==========")

labels = []

for subject_name, subject_data in data.items():
    for trial_name, trial in subject_data.items():

        other = trial.get("other", None)

        if other is not None:
            labels.append(str(other))

        break  # Only inspect one trial per subject

print(Counter(labels))