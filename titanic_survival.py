'''
TITANIC SURVIVAL PREDICTION
Data Dictionary
Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.


'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ydata_profiling import ProfileReport



file_path = './../data/tested.csv'
data = pd.read_csv(file_path)
# print(data['Age'].unique())
# print(data['Cabin'].unique())
# print(data.columns)
# # count of cabins per passenger class
# print(data.groupby('Pclass')['Cabin'].count())
# # list those cabins classwise
# cabin_by_class = data.groupby('Pclass')['Cabin'].apply(lambda x: x.dropna().unique().tolist())
# print(cabin_by_class)
# print(data.columns)
# print(data.shape)
# print(data.head(10))
# print(data['Parch'].head(20))
# print(data.loc[:,['Parch','SibSp']].nunique())
# print(data.loc[:,['Parch','SibSp']])
# print(data['SibSp'].unique())
# print(data['Parch'].unique())
# print(data.loc[:,['Age','Cabin']].nunique())
# print(data.loc[:,['Age','Cabin']])



# TARGET - ensure missing ones dropped
data.dropna(axis=0, subset=['Survived'], inplace=True)
y = data.Survived

# print(data['Survived'].nunique())
# print(data['Survived'].unique())

#  HANDLE MISSING VALUES in Age and Cabin.
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Cabin'] = data['Cabin'].fillna('Unknown')

# FEATURE CREATION: Family size, Title from Name.
data["Title"] = data["Name"].str.extract(r",\s*([^\.]*)\.")
title_mapping = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty"
}

data["Title"] = data["Title"].replace(title_mapping)
# print(data.head(10))
# +1 is the passenger themself
data["Family Size"] = data["SibSp"] + data["Parch"] + 1
# print(data[["SibSp", "Parch", "Family Size"]].head(10)) 
# Generate pandas-profiling report
profile = ProfileReport(data, title="Titanic Dataset EDA Report", explorative=True)

# Save report as HTML
profile.to_file("titanic_pandas_profiling_report.html")

print("Pandas Profiling EDA report generated successfully!")

# PREDICTOR
X = data.drop(['Survived'],axis=1)

# TRAINING/VALIDATION DATA
X_train, X_valid, y_train, y_valid = train_test_split(X, y,random_state=1)

categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
# print(categorical_cols)
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]
# print(numerical_cols)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


'''
Evaluate models: Logistic Regression, Random Forest.
'''
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
def score_dataset(model_class, X_train, X_valid, y_train, y_valid, **model_kwargs):
    """
    Fits any model and returns its MAE score.

    Parameters
    ----------
    model_class : class
        The model class (e.g., RandomForestRegressor, LinearRegression, XGBRegressor)
    X_train, X_valid, y_train, y_valid : DataFrames/Series
        Training and validation data.
    **model_kwargs : keyword arguments
        Model-specific parameters (like n_estimators, random_state, etc.)

    Returns
    -------
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay since our target is discrete
    
    """
    model = model_class(**model_kwargs)  # instantiate the model 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)

    if hasattr(model, "predict_proba"):
        probs = pipeline.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, probs)
    else:
        roc_auc = None

    acc = accuracy_score(y_valid, preds)
    cm = confusion_matrix(y_valid, preds)
    report = classification_report(y_valid, preds)

    print(f"🔹 Accuracy: {acc:.3f}")
    if roc_auc is not None:
        print(f"🔹 ROC-AUC: {roc_auc:.3f}")
    print("🔹 Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return acc

print("Random Forest:")
score_dataset(RandomForestClassifier, X_train, X_valid, y_train, y_valid, n_estimators=100, random_state=42)

print("\nLogistic Regression:")
score_dataset(LogisticRegression, X_train, X_valid, y_train, y_valid, max_iter=1000)
