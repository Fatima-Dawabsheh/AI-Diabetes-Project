from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

app = Flask(__name__)
data1 = pd.read_csv('dataset1.csv', delimiter=',')
data2 = pd.read_csv('dataset2.csv', delimiter=',')

target_columns = {
    '1': 'Outcome',
    '2': 'Diabetes'
}
@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/page2', methods=['POST'])
def page2():
    dataset_choice = request.form.get('dataset')
    algorithm_choice = request.form.get('algorithm')

    if dataset_choice =='1':
        data = data1
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif dataset_choice =='2':
        data = data2
        features = ['Age', 'Gender', 'BMI', 'SBP', 'DBP', 'FPG', 'Chol', 'Tri', 'HDL', 'LDL',
                    'ALT', 'BUN', 'CCR', 'FFPG', 'smoking', 'drinking', 'family_histroy']
    else:
        return "Invalid dataset choice", 400

    target_column = target_columns.get(dataset_choice)
    if target_column not in data.columns:
        return f"Error: The '{target_column}' column is missing in the selected dataset.", 400

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    if algorithm_choice == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif algorithm_choice == 'NaiveBayes':
        model = GaussianNB()
    elif algorithm_choice == 'NeuralNetwork':
        model = MLPClassifier(random_state=1, max_iter=300)
    elif algorithm_choice == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return "Invalid algorithm choice", 400

    
    model.fit(X_train, y_train)
    start_time = time.time()
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = round(accuracy_score(y_test, y_pred), 5)
    precision = round(precision_score(y_test, y_pred), 5)
    recall = round(recall_score(y_test, y_pred), 5)
    f1 = round(f1_score(y_test, y_pred), 5)
    elapsed_time = round(time.time() - start_time, 5)
    # Render page2 with metrics
    return render_template('page2.html',
                           conf_matrix=conf_matrix,
                           time_elapsed=elapsed_time,
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1,
                           dataset=dataset_choice,
                           algorithm=algorithm_choice)

@app.route('/page3', methods=['POST'])
def page3():
    dataset_choice = request.form.get('dataset')
    algorithm_choice = request.form.get('algorithm')
    if dataset_choice == '1':
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif dataset_choice == '2':
        features = ['Age', 'Gender', 'BMI', 'SBP', 'DBP', 'FPG', 'Chol', 'Tri', 'HDL', 'LDL',
                    'ALT', 'BUN', 'CCR', 'FFPG', 'smoking', 'drinking', 'family_histroy']
    else:
        return "Invalid dataset choice", 400

    return render_template('page3.html',
                           features=features,
                           algorithm=algorithm_choice,
                           dataset=dataset_choice)

@app.route('/predict', methods=['POST'])
def predict():
    dataset_choice = request.form.get('dataset')
    algorithm_choice = request.form.get('algorithm')  
    featurevalues = {}
    for key, value in request.form.items():
        if key in ['dataset', 'algorithm']:
            continue
        if key == 'Gender':
            featurevalues[key] = int(value) # Male=1,Female=2
        elif key == 'smoking':
            featurevalues[key] = int(value) # No=3,Yes=2,Sometime=1
        elif key == 'drinking':
            featurevalues[key] = int(value) # No=3,Yes=2,Sometime=1
        elif key == 'family_histroy':
            featurevalues[key] = int(value) # No=0,Yes=1
        else:
            featurevalues[key] = float(value) # Other features

    if dataset_choice =='1':
        data = data1
    elif dataset_choice =='2':
        data = data2
    else:
        return "Invalid dataset", 400
    target_column = target_columns.get(dataset_choice)
    if target_column not in data.columns:
        return f"Error: The '{target_column}' column is missing in the selected dataset.", 400
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    if algorithm_choice =='DecisionTree':
        model = DecisionTreeClassifier()
    elif algorithm_choice == 'NaiveBayes':
        model = GaussianNB()
    elif algorithm_choice =='NeuralNetwork':
        model = MLPClassifier(random_state=1, max_iter=300)
    elif algorithm_choice =='RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return "Invalid algorithm choice", 400

    model.fit(X, y)

    userdata = pd.DataFrame([featurevalues])
    predic = model.predict(userdata)

    return render_template('page4.html', predic=predic[0])

if __name__ == '__main__':
    app.run(debug=True)
