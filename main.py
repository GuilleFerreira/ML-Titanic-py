import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')

# Information about the dataframe
df.info()

# Check for missing values using isna() function
missing_values = df.isna()

# Count missing values in each column
missing_count = missing_values.sum()

# Print count of missing values
print("Missing value counts:")
print(missing_count)
print("\n")

# ===========================================
#              Data preparation
# ===========================================

# Drop column with missing values
df = df.drop('Cabin', axis=1)

# Drop columns which are not useful
df = df.drop('Name', axis=1)
df = df.drop('Fare', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('PassengerId', axis=1)

# Fill missing values in Age column with mean value
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing values in Embarked column with most frequent value and change categorical column to numerical column
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
embarked_numerical = pd.get_dummies(df['Embarked'])
df = df.drop('Embarked',axis=1)
df = df.join(embarked_numerical)


# Convert categorical columns to numerical columns using get_dummies() function
sex_numerical = pd.get_dummies(df['Sex'])
df = df.drop('Sex', axis=1)
df = df.join(sex_numerical)

# Verify if all missing values are filled
missing_values = df.isna()
missing_count = missing_values.sum()
print("Missing value counts:")
print(missing_count)

# Check head of dataframe
print(df.head())

# ===========================================
#     Create testing and training dataset
# ===========================================

# Import train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Drop the target variable from the dataframe and assign it to X.
X = df.drop('Survived', axis=1)

# y variable will be assigned to survived.
y = df['Survived']

# Split the dataset into training and testing dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# ===========================================
#            Apply Random Forest
# ===========================================

#Imports the Random Forest Model.
from sklearn.ensemble import RandomForestClassifier

#Imports the accuracy score.
from sklearn.metrics import accuracy_score

# Set up of the Random Forest Model
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Get predictions
prediction = random_forest.predict(X_test)

# Print the accuracy score
print('Accuracy score =', accuracy_score(y_test, prediction))
