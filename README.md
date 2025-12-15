# Predicting goat behaviour from ear-tag accelerometer data
This repository includes the data and key information necessary to train and test a model to identify goat behaviour from xyz coordinates in ear-tag accelerometers. This README also serves as the project report, discussing the motivation for the project, necessary pre-requisites, methods, results, and conclusions.

## Motivation
Text

## Pre-requisites
In terminal:
```
uv add numpy pandas matplotlib scikit-learn
```

In Jupyter lab (or similar):
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
```

## Methods
Load in data
```
# Get file names of all 8 files
files = [
    "ID_21038.csv",
    "ID_21055.csv",
    "ID_21066.csv",
    "ID_21083.csv",
    "ID_21091.csv",
    "ID_21101.csv",
    "ID_21118.csv",
    "ID_21119.csv"
]

# Read them
df_list = [pd.read_csv(f) for f in files]

# Combine into one dataframe 'df'
df = pd.concat(df_list, ignore_index=True)

# Check
print(df.shape)
df.head()
```

Extract only columns of interest: time, x coordinate, y coordinate, z coordinate, and the goat's behaviour ('position behaviour')
```
df = df[["TIME", "ACCx", "ACCy", "ACCz", "position_behav_data_goat"]]
df.head()
```

Clean the data
```
# Get rid of rows with NA
df = df.dropna()

# List behaviours identified
unique_values = df["position_behav_data_goat"].unique()
print(unique_values)

# Get rid of rows where behaviour is not visible
df = df.drop(df[df["position_behav_data_goat"] == "nvisible"].index)

# Combine lying and lyingd, and standing and standingp
label_map = {
    'lying': 'lying',
    'lyingd': 'lying',
    'standing': 'standing',
    'standingp': 'standing',
    'walking': 'walking',
    'milking': 'milking',
    'climbing': 'climbing',
}

# And apply
df["position_behav_data_goat"] = df["position_behav_data_goat"].map(label_map)

# Check it worked
unique_values = df["position_behav_data_goat"].unique()
print(unique_values)

# Check relative frequencies of categories
df['position_behav_data_goat'].value_counts()

# Drop climbing due to few occurences
df = df.drop(df[df["position_behav_data_goat"] == "climbing"].index)

# Check it worked
unique_values = df["position_behav_data_goat"].unique()
print(unique_values)
```

Wrangle date/time data
```
# Format "TIME" column as date time
df["TIME"] = pd.to_datetime(df["TIME"])

# Extract hour (so we can account for behavioural differences during time of day)
df["hour"] = df["TIME"].dt.hour

# Calculate time since start, in seconds
start_time = df["TIME"].iloc[0] # First set the initial time as 0 based on first value
df["time_seconds"] = (df["TIME"] - start_time).dt.total_seconds() # Then calculate time from there

# Calculate time elapsed since last measurement
df["delta_time"] = df["time_seconds"].diff().fillna(1e-6)  # Have this (1e-6) to avoid dividing by zero for the first row

# Check it has worked
df.head()

# Calculate change in x/y/z since last measurement
df["dACCx"] = df["ACCx"].diff().fillna(0)
df["dACCy"] = df["ACCy"].diff().fillna(0)
df["dACCz"] = df["ACCz"].diff().fillna(0)

# Check it has worked
df.head()
```

Calculate speed
```
# Calculate overall movement as magnitude of x/y/z change
df["ACC_mag_change"] = np.sqrt(df["dACCx"]**2 + df["dACCy"]**2 + df["dACCz"]**2)

# Calculate speed in units per second
df["speed"] = df["ACC_mag_change"] / df["delta_time"]

# Check it has worked
df.head()
```

Visually inspect the distribution of variables using plt. hist, for example:
```
plt.hist(df["dACCx"], color='skyblue', edgecolor='black')
```
Showed that while the distrubution of the coordinates could be log-transformed to be normal, this was not the case for speed. As such, a random forest model (which does not require a normal distribution) was chosen.

Set up random forest model
```
# Data for model: features and target
features = ["dACCx", "dACCy", "dACCz", "speed"]
X = df[features]
y = df["position_behav_data_goat"]

# Train/test split as 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the random forest
rf = RandomForestClassifier(
    n_estimators=100,      
    max_depth=30,        
    random_state=42,
    n_jobs=-1              
)
```

Train the model
```
# Fit to training data
rf.fit(X_train, y_train)
```

Use model to generate predictions
```
# Predict
y_pred = rf.predict(X_test)

```

Several combinations of the hyper-parameters 'number of estimators' and 'maximum tree depth' were tweaked and the value which gave the best model accuracy, whilst maintaining performance across the different behaviours, was selected.


## Results
Assess the accuracy and composition of these predictions in a confusion matrix
```
# Find accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Draw confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['lying', 'standing', 'walking', 'milking'])
disp.plot(cmap='Blues')
plt.title('matrix')
plt.show()
```
Accuracy was found to be 0.4971, or 49.71% (4sf)

<img width="584" height="446" alt="Screenshot 2025-12-15 at 10 57 21" src="https://github.com/user-attachments/assets/d20c54fa-8297-487b-b0a9-64f8f7b35305" />


Find feature importances
```
# Find feature importances
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances, align="center")
plt.xticks(range(X.shape[1]), features, rotation=90)
plt.show()
```
<img width="830" height="562" alt="Screenshot 2025-12-15 at 10 57 32" src="https://github.com/user-attachments/assets/bb1fee8f-c12c-4c08-bfc4-46025fe13d0b" />

## Conclusions
