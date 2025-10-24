"""
Streamlit application for spam email classification.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

# 預設的測試郵件範例
SAMPLE_EMAILS = {
    "正常郵件1": "Dear valued customer, Your account statement is now available. Please log in to your secure account to view it.",
    "正常郵件2": "Team meeting scheduled for tomorrow at 10 AM. Please prepare your weekly progress report.",
    "垃圾郵件1": "CONGRATULATIONS! You've won $1,000,000 in our lottery! Click here to claim your prize now!!!",
    "垃圾郵件2": "Buy now! 90% OFF on luxury watches! Limited time offer! Don't miss out!!!",
}

# 載入訓練數據
@st.cache_data
def load_training_data():
    # 這裡可以加入更多訓練數據
    emails = [
        "Your account statement is ready",
        "Meeting tomorrow at 10 AM",
        "Win million dollars now!!!",
        "90% OFF luxury watches!!!"
    ]
    labels = [0, 0, 1, 1]  # 0: 正常郵件, 1: 垃圾郵件
    return emails, labels

def train_model(emails, labels):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(emails)
    model = MultinomialNB()
    model.fit(X, labels)
    return vectorizer, model

def plot_probability_bar(probability):
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['green' if probability < 0.5 else 'red']
    sns.barplot(x=[probability], y=['Spam Probability'], ax=ax, palette=colors)
    ax.set_xlim(0, 1)
    plt.title('垃圾郵件機率')
    st.pyplot(fig)
    plt.close()

def main():
    st.title('垃圾郵件分類系統')
    
    # 載入和訓練模型
    emails, labels = load_training_data()
    vectorizer, model = train_model(emails, labels)
    
    # 選擇測試方式
    test_method = st.radio(
        "選擇測試方式",
        ["使用預設範例", "輸入自定義文本"]
    )
    
    if test_method == "使用預設範例":
        email_text = st.selectbox(
            "選擇一個測試郵件",
            list(SAMPLE_EMAILS.keys())
        )
        if email_text:
            text_to_analyze = SAMPLE_EMAILS[email_text]
            st.text_area("郵件內容", text_to_analyze, height=100)
    else:
        text_to_analyze = st.text_area(
            "輸入要測試的郵件內容",
            height=100
        )
    
    if st.button('開始分析') and text_to_analyze:
        # 特徵提取和預測
        X_test = vectorizer.transform([text_to_analyze])
        spam_prob = model.predict_proba(X_test)[0][1]
        
        # 顯示結果
        st.header('分析結果')
        result = "垃圾郵件" if spam_prob > 0.5 else "正常郵件"
        st.write(f"這封郵件很可能是: **{result}**")
        
        # 顯示機率條形圖
        plot_probability_bar(spam_prob)
        
        # 顯示詳細機率
        st.write(f"- 是正常郵件的機率: {(1-spam_prob)*100:.2f}%")
        st.write(f"- 是垃圾郵件的機率: {spam_prob*100:.2f}%")
        
        # 顯示重要特徵
        # 獲取當前文本中的詞彙
        current_text_features = vectorizer.transform([text_to_analyze])
        current_words = set()
        for idx, val in enumerate(current_text_features.toarray()[0]):
            if val > 0:
                current_words.add(vectorizer.get_feature_names_out()[idx])
        
        # 計算特徵重要性
        feature_importance = pd.DataFrame({
            'word': vectorizer.get_feature_names_out(),
            'importance': model.feature_log_prob_[1] - model.feature_log_prob_[0]
        })
        
        # 只保留當前文本中出現的詞
        feature_importance = feature_importance[feature_importance['word'].isin(current_words)]
        top_features = feature_importance.nlargest(5, 'importance')
        
        st.subheader('當前郵件中影響判斷的關鍵字')
        if not top_features.empty:
            for _, row in top_features.iterrows():
                st.write(f"- {row['word']}: {row['importance']:.4f}")
        else:
            st.write("沒有找到顯著影響判斷的關鍵字")

if __name__ == '__main__':
    main()