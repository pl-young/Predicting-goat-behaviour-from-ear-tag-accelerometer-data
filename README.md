# Predicting positional goat behaviour from ear-tag accelerometer data
This repository includes the data and key information necessary to train and test a model to identify goat behaviour from xyz coordinates in ear-tag accelerometers. This README also serves as the project report, discussing the motivation for the project, necessary pre-requisites to run the code, methods, results, and conclusions.

## Introduction and motivation
Animal welfare is of social, economic, environmental, and ethical importance (WOAH, 2024) and efforts to measure and improve welfare are therefore of value. In the past, measures of animal welfare have often focused on physical health. However, it is now accepted that animal welfare goes beyond physical health to encompass psychological health and consideration of the animal's emotional state (Mellor, 2020). Behaviour is suggested to be a useful tool to measure these aspects (Dawkins, 2003), as well as offering early insights into physical health concerns such as fatigue (Darbandi et al., 2023) and changes in gait (Bucci et al., 2025).

However, manual measurements of behaviour can be time-consuming, expensive, and unreliable (Rushen et al., 2012). Automated measurements, such as those gathered from accelerometers worn by each individual animal, represent a way to collect large amounts of behavioural data at a lower cost and with less time input. These data can then be used to train a model to predict the animal's behaviour from the accelerometer data. This model can then be employed in real-time to cheaply and easily monitor each animal's behaviour, potentially allowing the farmer to detect early signs of poor health and wellbeing and improve them before the animal's welfare is substantially compromised.

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
In this work I look at data from Mauny et al. (2025). In this study, accelerometers were attached to the ear-tags of eight indoor dairy Alpine goats to record their acceleration on the x-y-z-axis for a 24-hour period. Videos were also taken and the behaviours performed by each goat were labelled for a total of 11 hours per goat. These labels were made by a single trained observer and represented the 'ground truth'. Each file represents data from one individual goat.

The data were cleaned to remove NAs, entries where the behaviour was not visible, and behaviours with very few entries, and the variables of interest – x-y-z coordinates, the animal's behaviour, and time stamps – were extracted.

The x-y-z coordinates at each recording was used to calculate the animal’s overall movement as magnitude in the following way: sqrt[(change in x-coordinate since last recording^^2) + (change in y-coordinate since last recording^^2) + (change in z-coordinate since last recording^^2)]. Speed was then calculated as: magnitude / time since last recording.

The distribution of each variable to be included in the model – change in x-y-z coordinates since last measurement, and speed – was visually assessed in order to select an appropriate model. The model was trained on 75% of the data, leaving 25% for testing.

After generating an initial model, several of the hyper-parameters were tweaked and the hyper-parameter values which gave the best model accuracy, whilst maintaining performance across the different behaviours, were selected.


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
plt.xlabel("dACCx")
plt.ylabel("Frequency")
plt.show()
```
<img width="619" height="434" alt="Screenshot 2025-12-15 at 16 17 35" src="https://github.com/user-attachments/assets/aebb8472-ef33-4059-bfbd-e73d30847054" />

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


## Results
Transformation and visualisation of the variables to be included in the model showed that while the distrubution of the coordinates could be log-transformed to be normal, this was not the case for speed. As such, a random forest model (which does not require a normal distribution) was chosen.

After tweaking the hyper-parameters 'number of estimators' to be 100 and 'maximum tree depth' to be 30, the accuracy of the final model was 0.4971, or 49.71% (4sf). This is visualised in the confusion matrix.


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

<img width="584" height="446" alt="Screenshot 2025-12-15 at 10 57 21" src="https://github.com/user-attachments/assets/d20c54fa-8297-487b-b0a9-64f8f7b35305" />

Speed was the most important feature used in the model's predictions, but the x-y-z coordinates were also used.

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
At 49.71%, the accuracy of the final model was poor. This indicates that the was model was not reliably able to predict the behaviour of an individual goat using accelerometer data.

This analysis does face some limitations. Data were only gathered for eight individual goats over one 24-hour period, and video data was only labelled for a total fo 11 hours per goat. This small sample size could have limited the ability of the model to accurately predict behaviour. A larger sample may have allowed the model to make more accurate predictions, and increased the generalisability of these predictions to the wider population. Despite the observer being trained, it is also possible that they made some errors during the manual labelling of the videos. However, this is unlikely to have caused the substantial degree of inaccuracy seen in this model. Finally, computational limitations prevented thorough optimisation of hyper-parameters, and it is possible that further tweaking could have increased model accuracy to some extent.

More fundamentally, however, these results indicate that a random forest model trained on x-y-z coordinate and time stamp data from accelerometers in the method described above is not a reliable predictor of goat behaviour. Further work on these data may wish to use another method to calculate movement, or may be able to more effectively optimise the model, in order to make a more thorough assessment of whether accelerometer data can be effectively used to predict goat behaviour.

## References
Bucci, M. P., Dewberry, L. S., Staiger, E. A., Allen, K., & Brooks, S. A. (2025). AI-assisted Digital Video Analysis Reveals Changes in Gait Among Three-Day Event Horses During Competition. Journal of Equine Veterinary Science, 105344. https://doi.org/10.1016/j.jevs.2025.105344

Darbandi, H., Munsters, C. C. B. M., Parmentier, J., & Havinga, P. (2023). Detecting fatigue of sport horses with biomechanical gait features using inertial sensors. PLOS ONE, 18(4), e0284554–e0284554. https://doi.org/10.1371/journal.pone.0284554

Dawkins, M. S. (2003). Behaviour as a tool in the assessment of animal welfare. Zoology, 106(4), 383–387. https://doi.org/10.1078/0944-2006-00122

Mauny, S., Kwon, J., Friggens, N. C., Duvaux-Ponter, C., & Taghipoor, M. (2025). Data paper: A goat behaviour dataset combining labelled behaviours and accelerometer data for training Machine Learning detection models. Animal - Open Space, 4, 100095. https://doi.org/10.1016/j.anopes.2025.100095

Mellor, D. J., Beausoleil, N. J., Littlewood, K. E., McLean, A. N., McGreevy, P. D., Jones, B., & Wilkins, C. (2020). The 2020 five domains model: Including human–animal interactions in assessments of animal welfare. Animals, 10(10), 1870. National Library of Medicine. https://doi.org/10.3390/ani10101870

Rushen, J., Chapinal, N., & de Passillé, A. (2012). Automated monitoring of behavioural-based animal welfare indicators. Animal Welfare, 21(3), 339–350. https://doi.org/10.7120/09627286.21.3.339

World Organisation for Animal Health. (2024). WOAH Vision Paper - Animal welfare: a vital asset for a more sustainable world. https://doi.org/10.20506/woah.3440
