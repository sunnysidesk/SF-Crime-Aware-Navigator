from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import pandas as pd

def prepare_model_data(df, day_df):
    model_df = pd.concat([
        df[['incident_hour', 'incident_minute', 'latitude', 'longitude', 'risk_level', 'weight']],
        day_df
    ], axis=1)
    X = model_df.drop(['risk_level', 'weight'], axis=1)
    y = model_df['risk_level']
    weights = model_df['weight']
    return X, y, weights

def split_data(X, y, weights, test_size=0.2, random_state=42):
    return train_test_split(X, y, weights, stratify=y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, w_train):
    clf = RandomForestClassifier(class_weight=None, random_state=42)
    clf.fit(X_train, y_train, sample_weight=w_train)
    return clf

def evaluate_model(clf, X_test, y_test, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    y_probs = clf.predict_proba(X_test)[:, 1]
    results = []
    for t in thresholds:
        y_pred_thresh = (y_probs > t).astype(int)
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        results.append({'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1})
    return results

def save_model(clf, ohe, model_path='models/risk_model.joblib', ohe_path='models/encoder.joblib'):
    joblib.dump(clf, model_path)
    joblib.dump(ohe, ohe_path)

def load_model(model_path='models/risk_model.joblib', ohe_path='models/encoder.joblib'):
    clf = joblib.load(model_path)
    ohe = joblib.load(ohe_path)
    return clf, ohe