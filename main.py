import os
import re
import numpy as np
from collections import Counter
from math import log

# 停用詞列表
STOP_WORDS = {'the', 'and', 'is', 'are', 'was', 'were', 'to', 'of', 'a', 'an', 
              'it', 'this', 'that', 'on', 'for', 'in', 'at', 'with', 'from', 'by', 
              'about', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'not', 'so', 
              'am', 'i', 'you', 'we', 'he', 'she', 'they', 'their', 'our', 'my', 'your'}

# 文檔前處理函數
def preprocess_text(text):
    # 移除數字、標點符號並轉小寫
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    # 移除停用詞
    words = [word for word in words if word not in STOP_WORDS]
    return words

# 建立詞典
def build_vocabulary(documents):
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc)
    word_to_index = {word: idx for idx, word in enumerate(sorted(vocabulary))}
    return word_to_index

# 建立詞頻矩陣
def build_term_frequency_matrix(documents, vocabulary):
    num_docs = len(documents)
    num_words = len(vocabulary)
    tf_matrix = np.zeros((num_docs, num_words), dtype=int)
    
    for doc_idx, doc in enumerate(documents):
        word_counts = Counter(doc)
        for word, count in word_counts.items():
            if word in vocabulary:
                word_idx = vocabulary[word]
                tf_matrix[doc_idx][word_idx] = count
    return tf_matrix

# 計算卡方統計量
def calculate_chi_square(tf_matrix, labels):
    num_docs, num_words = tf_matrix.shape
    chi_square_scores = np.zeros(num_words)
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    for word_idx in range(num_words):
        A = sum(tf_matrix[doc_idx][word_idx] for doc_idx in range(num_docs) if labels[doc_idx] == 1)
        B = sum(tf_matrix[doc_idx][word_idx] for doc_idx in range(num_docs) if labels[doc_idx] == 0)
        C = total_pos - A
        D = total_neg - B
        numerator = (A * D - B * C) ** 2
        denominator = (A + C) * (B + D) * (A + B) * (C + D)
        if denominator > 0:
            chi_square_scores[word_idx] = numerator / denominator
    return chi_square_scores

# 主程序
if __name__ == "__main__":
    # 要求輸入人名
    person_name = input("請輸入人名 :").strip()
    folder_path = f"./{person_name}"  # 動態設定資料夾路徑

    # 確保資料夾存在
    if not os.path.exists(folder_path):
        print(f"錯誤：資料夾 '{folder_path}' 不存在。請確認路徑。")
        exit()

    # 讀取指定資料夾下的文本檔案
    print(f"正在讀取 '{folder_path}' 資料夾中的文本...")
    target_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8', errors='ignore') as file:
                target_docs.append(preprocess_text(file.read()))

    # 背景文本 (這裡使用空白背景，你可加入其他文本作為背景)
    background_docs = [[""]]
    all_docs = target_docs + background_docs
    labels = [1] * len(target_docs) + [0] * len(background_docs)
    
    # 建立詞典和詞頻矩陣
    vocabulary = build_vocabulary(all_docs)
    tf_matrix = build_term_frequency_matrix(all_docs, vocabulary)
    
    # 計算卡方分數
    print("正在計算卡方統計量...")
    chi_square_scores = calculate_chi_square(tf_matrix, labels)
    sorted_indices = np.argsort(-chi_square_scores)[:300]  # 取前 300 個分數最高的詞

    # 輸出結果
    output_file = f"{person_name}_top300.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in sorted_indices:
            word = list(vocabulary.keys())[idx]
            f.write(f"{word}\n")
    
    print(f"特徵詞已保存至 '{output_file}'")
