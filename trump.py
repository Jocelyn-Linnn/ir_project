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

# 自訂停用詞
CUSTOM_STOP_WORDS = {'trump', 'sec', 'donald' ,'i', 'am', 'are', 'is', 'was', 'were', 'be', 'been', 
                     'has', 'have', 'had', 'do', 'does', 'did', 'the', 'a', 'an', 'in', 'on', 
                     'for', 'with', 'to', 'of', 'and', 'or', 'this', 'that', 'it', 'its', 'at', 
                     'by', 'as', 'from', 'will', 'can', 'cannot', 'would', 'could', 'should', 
                     'not', 'so', 'we', 'you', 'he', 'she', 'they', 'them', 'their', 'our', 'my'}

STOP_WORDS = DEFAULT_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# 文檔前處理函數
def preprocess_text(text):
    # 移除數字、標點符號，轉小寫
    text = re.sub(r'\d+', '', text)  # 移除數字
    text = re.sub(r'[^\w\s]', '', text.lower())  # 移除標點符號並轉為小寫
    words = text.split()
    # 移除停用詞並進行詞幹提取
    words = [STEMMER.stem(word) for word in words if word not in STOP_WORDS]
    return " ".join(words)

# 載入指定資料夾中的所有文檔
def load_documents(folder_path):
    documents = []
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
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
    # 載入特朗普演講文檔
    trump_folder = "./trump"
    trump_speeches = load_documents(trump_folder)
    
    # 這裡背景文檔設為空，因為未提供背景數據，可擴展為其他政客的文檔
    background_docs = [""]  # 如果有背景文本，填入這裡的資料
    
    # 提取特徵詞
    top_words = chi_square_feature_selection(trump_speeches, background_docs, top_n=300)
    
    # 儲存結果至檔案
    output_file = "trump_top300.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for word in top_words:
            f.write(f"{word}\n")
    
    print(f"特徵詞已保存至 {output_file}")
