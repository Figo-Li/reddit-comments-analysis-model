import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_data():
    train_data = pd.read_csv("./codabench_dataset/train_data.csv")
    test_data = pd.read_csv("./codabench_dataset/test_data.csv")
    val_data = pd.read_csv("./codabench_dataset/val_data.csv")
    
    return train_data, test_data, val_data

class logistic_regression_model():
    def __init__(self):
        self.train_data, self.test_data, self.val_data = load_data()
        
    def TFIDF_Comment(self):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        self.X_train = vectorizer.fit_transform(self.train_data['Comment'].fillna(''))
        self.X_test = vectorizer.transform(self.test_data['Comment'].fillna(''))
        self.X_val = vectorizer.transform(self.val_data['Comment'].fillna(''))
                
    def train(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
        self.model.fit(self.X_train, self.train_data['Ground_Truth_Label'])

    def predict(self):
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_val = self.model.predict(self.X_val)

    def show_performance(self): 
        print("Test Accuracy:", accuracy_score(self.test_data['Ground_Truth_Label'], self.y_pred_test))
        print("Validation Accuracy:", accuracy_score(self.val_data['Ground_Truth_Label'], self.y_pred_val))
        print("Classification Report on Test Data:\n", classification_report(self.test_data['Ground_Truth_Label'], self.y_pred_test))
        
class Random_Forest_Model():
    def __init__(self):
        self.train_data, self.test_data, self.val_data = load_data()
        
    def TFIDF_Comment(self):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
        self.X_train = vectorizer.fit_transform(self.train_data['Comment'].fillna(''))
        self.X_test = vectorizer.transform(self.test_data['Comment'].fillna(''))
        self.X_val = vectorizer.transform(self.val_data['Comment'].fillna(''))
                
    def train(self):
        self.model = RandomForestClassifier(class_weight='balanced')
        self.model.fit(self.X_train, self.train_data['Ground_Truth_Label'])

    def predict(self):
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_val = self.model.predict(self.X_val)

    def show_performance(self): 
        print("Test Accuracy:", accuracy_score(self.test_data['Ground_Truth_Label'], self.y_pred_test))
        print("Validation Accuracy:", accuracy_score(self.val_data['Ground_Truth_Label'], self.y_pred_val))
        print("Classification Report on Test Data:\n", classification_report(self.test_data['Ground_Truth_Label'], self.y_pred_test))

def main():
    lr_model = logistic_regression_model()
    lr_model.TFIDF_Comment()
    lr_model.train()
    lr_model.predict()
    lr_model.show_performance()
    
    rf_model = Random_Forest_Model()
    rf_model.TFIDF_Comment()
    rf_model.train()
    rf_model.predict()
    rf_model.show_performance()
    
main()