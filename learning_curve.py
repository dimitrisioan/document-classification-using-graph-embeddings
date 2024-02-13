import time
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from useful_methods import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

start_time = time.time()

parsed_path, prefix, choice = choose_dataset()
load_save_path = load_save_results(prefix, choice)
# parsed_path = "datasets_2/20newsgroups/newsgroups_dataset_parsed/"

def plot_learning_curve(estimator, X, y, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')

    plt.grid()
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r'
    )
    plt.fill_between(
        train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g'
    )
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation Score')

    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # Load the CSV file for Word2Vec from the corresponding dataset directory
    df = pd.read_csv(os.path.join(load_save_path, f'{prefix}_embeddings_word2vec.csv'))
    # df = pd.read_csv('all_categories_word2vec.csv')

    # Convert the embeddings column from string to list of floats
    X = df['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=',')).tolist()
    y = df['category']

    class_names = sorted(y.unique())
    print(class_names)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train MLP classifier
    model_mlp = MLPClassifier(early_stopping=True)
    model_mlp.fit(X_train, y_train)

    # Predict the categories of the test data
    y_pred = model_mlp.predict(X_test)

    # Evaluate MLP classifier
    acc_score_mlp = accuracy_score(y_test, y_pred)
    prec_score_mlp = precision_score(y_test, y_pred, average='weighted')
    conf_matrix_mlp = confusion_matrix(y_test, y_pred)
    report_mlp = classification_report(y_test, y_pred)

    print("MLP classifier:\n")
    print(f"Accuracy: {acc_score_mlp}")
    print(f"Precision: {prec_score_mlp}")
    print(f"Confusion matrix:\n {conf_matrix_mlp}")
    print(f"Report:\n {report_mlp}")

    plot_learning_curve(model_mlp, X, y)

    # Evaluate performance on the training set
    train_predictions = model_mlp.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Training Accuracy:", train_accuracy)

    # Evaluate performance on the testing set
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Testing Accuracy:", test_accuracy)

    print("--- %s seconds ---" % (time.time() - start_time))
