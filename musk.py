import os
import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer

# 初始化工具
STEMMER = PorterStemmer()
DEFAULT_STOP_WORDS = set(stopwords.words('english'))

# 自訂停用詞（針對 Elon Musk）
CUSTOM_STOP_WORDS = {'elon', 'musk', 'the', 'and', 'is', 'are', 'was',
                     'were', 'to', 'of', 'a', 'an', 'it', 'this', 'that', 'on', 'for', 'in', 'at', 
                     'with', 'from', 'by', 'about', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'can', 'cannot', 'so', 'not', 'am', 'i', 'you', 'we', 'he', 'she', 'they',
                     'their', 'our', 'my', 'your'}
STOP_WORDS = DEFAULT_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# 文檔前處理函數
def preprocess_text(text):
    # 移除數字和標點符號，轉小寫
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    return " ".join(words)

# 載入所有 .txt 檔案
def load_documents(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # 確保只讀取 txt 檔案
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                processed_content = preprocess_text(content)
                documents.append(processed_content)
    return documents

# 使用卡方檢定選出特徵詞
def chi_square_feature_selection(target_docs, background_docs, top_n=300):
    all_docs = target_docs + background_docs
    labels = [1] * len(target_docs) + [0] * len(background_docs)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_docs)
    y = np.array(labels)
    
    chi_scores, _ = chi2(X, y)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = feature_names[np.argsort(-chi_scores)[:top_n]]
    return top_features

# 主程序
if __name__ == "__main__":
    # 載入 Elon Musk 的逐字稿
    musk_folder = "./Musk"
    musk_docs = load_documents(musk_folder)
    
    # 這裡背景文檔設為空，未提供背景資料可填補其他相關人物文檔
    background_docs = [""]  # 若有背景資料，請補充在這裡
    
    # 提取特徵詞
    top_words = chi_square_feature_selection(musk_docs, background_docs, top_n=300)
    
    # 儲存結果至檔案
    output_file = "musk_top300.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for word in top_words:
            f.write(f"{word}\n")
    
    print(f"Elon Musk 特徵詞已保存至 {output_file}")
