# دورة حياة مشروع تحليل المشاعر

## 1. تحميل البيانات (Data Loading)
```python
def load_data():
    # تحميل البيانات من ملف CSV
    df = pd.read_csv('SmallReviews.csv')
    print(f"Loaded {len(df)} reviews")
    return df
```
- يتم تحميل البيانات من ملف `SmallReviews.csv`
- طباعة عدد المراجعات التي تم تحميلها
- إرجاع DataFrame يحتوي على البيانات

## 2. معالجة البيانات (Data Preprocessing)
```python
def preprocess_text(text):
    if isinstance(text, str):
        # تحويل النص إلى أحرف صغيرة
        text = text.lower()
        
        # إزالة علامات الترقيم
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # إزالة الأرقام
        text = re.sub(r'\d+', '', text)
        
        # تقسيم النص إلى كلمات
        tokens = word_tokenize(text)
        
        # إزالة الكلمات الوظيفية
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # معالجة الكلمات السلبية
        tokens = handle_negation(tokens)
        
        # معالجة كلمات التأكيد
        tokens = handle_intensifiers(tokens)
        
        # تصريف الكلمات
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        
        # تصحيح الكلمات
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    return ''
```
- تحويل النص إلى أحرف صغيرة
- إزالة علامات الترقيم والأرقام
- تقسيم النص إلى كلمات
- إزالة الكلمات الوظيفية
- معالجة الكلمات السلبية والتأكيد
- تصريف وتصحيح الكلمات

## 3. تحضير البيانات للتدريب (Data Preparation)
```python
def prepare_data(df):
    # تحويل التقييمات إلى مشاعر ثنائية
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    
    # معالجة النصوص
    df['Text'] = df['Text'].apply(preprocess_text)
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        df['Text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    return X_train, X_test, y_train, y_test
```
- تحويل التقييمات إلى مشاعر (إيجابي/سلبي)
- معالجة النصوص
- تقسيم البيانات إلى مجموعات تدريب واختبار

## 4. تدريب النموذج (Model Training)
```python
def train_model(X_train, y_train):
    # إنشاء pipeline
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
    
    return pipeline
```
- إنشاء pipeline يتضمن:
  - تحويل النصوص إلى متجهات TF-IDF
  - معالجة عدم توازن البيانات باستخدام SMOTE
  - تحجيم الميزات
  - نموذج الانحدار اللوجستي

## 5. تقييم النموذج (Model Evaluation)
```python
def evaluate_model(model, X_test, y_test):
    # التنبؤ
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # طباعة مقاييس التقييم
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # إنشاء مصفوفة الارتباك
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('confusion_matrix.png')
    
    # إنشاء منحنى ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.savefig('roc_curve.png')
```
- التنبؤ على بيانات الاختبار
- حساب دقة النموذج
- إنشاء تقرير التصنيف
- إنشاء مصفوفة الارتباك
- إنشاء منحنى ROC

## 6. حفظ النموذج (Model Saving)
```python
def save_model(model):
    # حفظ النموذج
    joblib.dump(model, 'sentiment_model.joblib')
    print("Model saved successfully!")
```
- حفظ النموذج المدرب للاستخدام المستقبلي

## 7. واجهة المستخدم (User Interface)
```python
def main():
    st.title("تحليل المشاعر")
    
    # تحميل النموذج
    model = load_model()
    
    # واجهة المستخدم
    text = st.text_area("أدخل النص هنا:")
    if st.button("تحليل"):
        if text:
            # التنبؤ
            prediction = predict_sentiment(text, model)
            st.write(f"النتيجة: {prediction}")
```
- إنشاء واجهة مستخدم باستخدام Streamlit
- تحميل النموذج المدرب
- السماح للمستخدم بإدخال نص
- عرض نتيجة التحليل

## 8. تشغيل التطبيق (Application Deployment)
```bash
streamlit run app.py
```
- تشغيل التطبيق على المتصفح
- السماح للمستخدمين بتجربة النموذج

## ملاحظات هامة:
1. **معالجة البيانات:**
   - يجب تنظيف البيانات جيداً
   - معالجة القيم المفقودة
   - معالجة عدم توازن البيانات

2. **تحسين النموذج:**
   - تجربة خوارزميات مختلفة
   - ضبط المعلمات
   - استخدام تقنيات التحقق المتقاطع

3. **تقييم النموذج:**
   - استخدام مقاييس متعددة
   - تحليل الأخطاء
   - تحسين الأداء

4. **نشر التطبيق:**
   - التأكد من جاهزية التطبيق
   - اختبار الواجهة
   - جمع التغذية الراجعة 