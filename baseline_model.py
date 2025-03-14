import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def load_data():
    train_data = pd.read_csv("./codabench_dataset/train_data.csv")
    test_data = pd.read_csv("./codabench_dataset/test_data.csv")
    val_data = pd.read_csv("./codabench_dataset/val_data.csv")
    
    return train_data, test_data, val_data

def main():
    train_data, test_data, val_data = load_data()
    np.random.seed(42)
    random_preds = np.random.choice(train_data['Ground_Truth_Label'].unique(), size=len(test_data))

    print("=== Random Baseline Classification Report ===")
    print(classification_report(test_data['Ground_Truth_Label'], random_preds))
    
main()
