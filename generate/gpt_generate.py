# apikey : 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


import os
from openai import OpenAI
from dotenv import load_dotenv
import sys
import io

# 確保目錄存在
output_dir = "Term_Project/result"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "mars.txt")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 載入 .env 文件中的 API 金鑰
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=api_key)

# 讀取名人特徵檔案
def load_character_features(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 名人特徵檔案路徑
files = {
    "Elon Musk": "Term_Project/Musk.txt",
    "Daniel Trump": "Term_Project/trump.txt",
    "William Shakespeare": "Term_Project/Shakespeare.txt",
    "Kevin Hart": "Term_Project/Hart.txt",
}

# 載入特徵內容
features = {name: load_character_features(path) for name, path in files.items()}

# 設定對話起始訊息
topic = "Should humanity migrate to Mars?"
messages = [
    {"role": "system", "content": "You are an expert language model that simulates interactive conversations between specific individuals based on their speech characteristics."},
]

# 準備起始提示
initial_prompt = f"Four celebrities are discussing the topic: '{topic}'. Their characteristics are as follows:\n"
for name, characteristic in features.items():
    initial_prompt += f"\n{name}:\n{characteristic}\n"

initial_prompt += "\nThey will now begin a discussion. Start the conversation."

# 添加起始對話訊息
messages.append({"role": "user", "content": initial_prompt})

# 設定生成多輪對話
conversation_turns = 3  # 設定互動回合數
# 打開文件以追加模式寫入
with open(output_file, "w", encoding="utf-8") as file:
    for _ in range(conversation_turns):
        # 呼叫模型生成新回應
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # 確認你的模型名稱正確
            messages=messages
        )
        
        # 獲取生成的內容
        response = completion.choices[0].message.content
        print(response)  # 打印到終端（或儲存到檔案中）

        # 將生成內容添加到對話中
        messages.append({"role": "assistant", "content": response})

        # 寫入文件
        file.write(response + "\n\n")
