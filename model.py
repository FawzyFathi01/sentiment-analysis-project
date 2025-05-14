import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load the preprocessed data
def load_data():
    print("Loading data...")
    df = pd.read_csv('SmallReviews.csv')
    print(f"Loaded {len(df)} reviews")
    return df

def preprocess_text(text):
    # تحويل النص إلى أحرف صغيرة
    text = text.lower()
    # إزالة علامات الترقيم والأرقام
    text = re.sub(r'[^\w\s]', '', text)
    # إزالة الأرقام
    text = re.sub(r'\d+', '', text)
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(df):
    print("Preparing data...")
    # تحويل التقييمات إلى مشاعر ثنائية
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    
    # معالجة النصوص
    print("Preprocessing texts...")
    df['Text'] = df['Text'].apply(preprocess_text)
    
    # طباعة توزيع الفئات
    print("\nClass distribution before balancing:")
    print(df['sentiment'].value_counts(normalize=True))
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("\nTraining model...")
    
    # إنشاء pipeline مع SMOTE
    pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=40000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3),
            sublinear_tf=True,
            use_idf=True,
            stop_words='english'
        )),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(
            max_iter=3000,
            C=1.0,
            class_weight='balanced',
            solver='saga',
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    # تدريب النموذج
    pipeline.fit(X_train, y_train)
    
    # الحصول على المكونات
    vectorizer = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['clf']
    
    # طباعة أهم الميزات
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    top_positive = np.argsort(coefficients)[-10:]
    top_negative = np.argsort(coefficients)[:10]
    
    print("\nTop positive features:")
    for i in top_positive:
        print(f"{feature_names[i]}: {coefficients[i]:.3f}")
    
    print("\nTop negative features:")
    for i in top_negative:
        print(f"{feature_names[i]}: {coefficients[i]:.3f}")
    
    return model, vectorizer, pipeline

def evaluate_model(model, vectorizer, pipeline, X_test, y_test):
    print("\nEvaluating model...")
    
    # التنبؤ
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # طباعة مقاييس التقييم
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # إنشاء مصفوفة الارتباك
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # إنشاء منحنى ROC
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # طباعة أمثلة للتنبؤات
    print("\nExample Predictions:")
    for i in range(min(5, len(X_test))):
        print(f"\nText: {X_test.iloc[i][:100]}...")
        print(f"True: {'Positive' if y_test.iloc[i] == 1 else 'Negative'}")
        print(f"Predicted: {'Positive' if y_pred[i] == 1 else 'Negative'}")
        print(f"Confidence: {y_pred_proba[i]:.2%}")

def save_model(model, vectorizer, pipeline):
    print("\nSaving model...")
    joblib.dump(model, 'sentiment_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(pipeline, 'sentiment_pipeline.joblib')
    print("Model, vectorizer, and pipeline saved successfully!")

def main():
    # تحميل البيانات
    df = load_data()
    
    # تحضير البيانات
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # تدريب النموذج
    model, vectorizer, pipeline = train_model(X_train, y_train)
    
    # تقييم النموذج
    evaluate_model(model, vectorizer, pipeline, X_test, y_test)
    
    # حفظ النموذج
    save_model(model, vectorizer, pipeline)

if __name__ == "__main__":
    main() 