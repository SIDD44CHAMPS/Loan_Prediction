import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Read file
data = pd.read_csv("loan_data_set.xls.csv")

# Split single column into many
data = data.iloc[:, 0].str.split(",", expand=True)

data.columns = [
    "Loan_ID","Gender","Married","Dependents","Education",
    "Self_Employed","ApplicantIncome","CoapplicantIncome",
    "LoanAmount","Loan_Amount_Term","Credit_History",
    "Property_Area","Loan_Status"
]

# Replace empty string with NaN
data.replace("", np.nan, inplace=True)

# Drop Loan_ID (not needed)
data.drop("Loan_ID", axis=1, inplace=True)

# Convert numeric columns
num_cols = [
    "ApplicantIncome","CoapplicantIncome",
    "LoanAmount","Loan_Amount_Term","Credit_History"
]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")

# Convert Dependents
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce")

# Fill missing numeric values
data[num_cols + ["Dependents"]] = data[num_cols + ["Dependents"]].fillna(
    data[num_cols + ["Dependents"]].median()
)

# Fill missing categorical values using mode
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical values
data.replace({
    "Gender": {"Male":1, "Female":0},
    "Married": {"Yes":1, "No":0},
    "Education": {"Graduate":1, "Not Graduate":0},
    "Self_Employed": {"Yes":1, "No":0},
    "Loan_Status": {"Y":1, "N":0},
    "Property_Area": {"Urban":2, "Semiurban":1, "Rural":0}
}, inplace=True)

# Split X and y
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))

print("✅ Model trained & saved successfully")
