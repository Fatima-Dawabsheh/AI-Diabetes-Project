# all packages needed
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import time
import pandas as pd

# Load datasets into DataFrames
Dataset1 = pd.read_csv('dataset1.csv',delimiter=',')
Dataset2 = pd.read_csv('dataset2.csv',delimiter=',')

# target column 
t_columns = {
    '1': 'Outcome',
    '2': 'Diabetes'
}

print("Dataset1 : 2768")
print("Dataset2 : 4303 , This dataset has more number of features")
choice = input("Enter the dataset you want to test 1 or 2 : ")
if choice == '1':
    data = Dataset1
elif choice == '2':
    data = Dataset2
else:
    print("Invalid dataset")
    exit()

# Verify the target column exists
target_column = t_columns.get(choice)
if target_column not in data.columns:
    print(f"Error: The'{target_column}'column is missing in the selected dataset")
    print("Available columns:", data.columns)
    exit()

# Separate features (X) and labels (y)
X = data.drop(target_column, axis=1)
y = data[target_column]
# Split the dataset into 80% training and 20% testing sets 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

while True:
    print("\t")
    print("Choose the algorithm :")
    print("1.Decision Tree")
    print("2.Naive Bayes")
    print("3.Neural Network")
    print("4.Random Forest")
    print("5.Exit")
    choice = input("Enter your choice 1, 2, 3, 4, or 5: ")

    if choice == '1':
        model = DecisionTreeClassifier()
    elif choice == '2':
        model = GaussianNB()
    elif choice == '3':
        model = MLPClassifier(random_state=1, max_iter=300)
    elif choice == '4':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif choice == '5':
        break
    else:
        print("Invalid choice")
        continue

    # Train the model
    model.fit(Xtrain, ytrain)

    # Start time
    start_time = time.time()

    # Make predictions on the test set
    ypred = model.predict(Xtest)

    # Calculate confusion matrix
    Conf_Matrix = confusion_matrix(ytest, ypred)

    # Calculate evaluation metrics
    Accuracy = accuracy_score(ytest, ypred)
    precision = precision_score(ytest, ypred)
    Recall = recall_score(ytest, ypred)
    F1 = f1_score(ytest, ypred)

    # Round to 5 decimal places
    Accuracy = round(Accuracy,5)
    precision = round(precision,5)
    Recall = round(Recall,5)
    F1 = round(F1,5)
    Time_elapsed = round(time.time()-start_time,5)

    # Print algorithm name
    if choice == '1':
        print("\nDecision Tree Algorithm:")
    elif choice == '2':
        print("\nNaive Bayes Algorithm:")
    elif choice == '3':
        print("\nNeural Network Algorithm:")
    elif choice == '4':
        print("\nRandom Forest Algorithm:")

    # Print confusion matrix
    print("Confusion Matrix:")
    print("\t\t\tActual Positive\t\tActual Negative")
    print("Classified Positive\t\t", Conf_Matrix[1 , 1],"\t\t", Conf_Matrix[1 , 0])
    print("Classified Negative\t\t", Conf_Matrix[0 , 1],"\t\t", Conf_Matrix[0 , 0])

    # Print evaluation metrics
    print("\n")
    print("Time elapsed :", Time_elapsed, "seconds")
    print("Accuracy :", Accuracy)
    print("Precision :", precision)
    print("Recall :", Recall)
    print("F1-score :", F1)
    print("\n")

    # Test for user 
    value1 = input(f"Do you want to test values from your own of this algorithm (y/n)? ")
    if value1.lower() =='y':
        features = {}
        for column in X.columns:
            value = input(f"Enter value for {column}: ")
            features[column]=[value]

        # Create a DataFrame 
        User_Data = pd.DataFrame(features)

        # Make predictions 
        User_Pred = model.predict(User_Data)

        # prediction
        if User_Pred[0] == 1:

            print("The expected result is to have diabetes")
        else:
            print("The expected result is not to have diabetes")
    else:
        continue