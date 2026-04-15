"""
Part 1: Bag of Words 情感分类

使用词袋模型 + 随机森林完成 IMDB 电影评论情感分析
"""

import pandas as pd
import numpy as np
import re
import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime


def review_to_words(raw_review):
    """将原始评论转换为干净的单词列表"""
    # 去除 HTML 标签
    markup = BeautifulSoup(raw_review, 'html.parser')
    markup_text = markup.get_text()
    
    # 去除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", markup_text)
    
    # 转为小写并分割
    words = letters_only.lower().split()
    
    return " ".join(words)


def main():
    # 数据路径
    DATA_DIR = "../../data"
    SUBMISSION_DIR = "../submissions"
    
    # 确保提交目录存在
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Part 1: Bag of Words 情感分类")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train = pd.read_csv(
        os.path.join(DATA_DIR, "labeledTrainData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    test = pd.read_csv(
        os.path.join(DATA_DIR, "testData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    print(f"   训练集：{train.shape}")
    print(f"   测试集：{test.shape}")
    
    # 2. 文本清洗
    print("\n2. 文本清洗...")
    print("   清洗训练数据...")
    clean_train_reviews = [review_to_words(review) for review in train['review']]
    
    print("   清洗测试数据...")
    clean_test_reviews = [review_to_words(review) for review in test['review']]
    print("   清洗完成!")
    
    # 3. 特征提取
    print("\n3. 词袋模型特征提取...")
    vectorizer = CountVectorizer(
        analyzer='word',
        max_features=5000,
        min_df=2,
    )
    
    train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()
    
    print(f"   训练特征：{train_data_features.shape}")
    print(f"   测试特征：{test_data_features.shape}")
    
    # 4. 训练模型
    print("\n4. 训练随机森林分类器...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_data_features, train['sentiment'],
        test_size=0.2, random_state=42
    )
    
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        verbose=0
    )
    
    forest.fit(X_train, y_train)
    
    # 5. 验证集评估
    y_pred = forest.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"   验证集准确率：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # 6. 全量训练并预测
    print("\n5. 全量训练并生成提交文件...")
    forest.fit(train_data_features, train['sentiment'])
    result = forest.predict(test_data_features)
    
    # 保存提交文件
    output = pd.DataFrame({
        'id': test['id'],
        'sentiment': result
    })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(SUBMISSION_DIR, f"bow_baseline_{timestamp}.csv")
    output.to_csv(submission_path, index=False, quoting=3)
    
    print(f"   提交文件：{submission_path}")
    print(f"   正面 (1): {sum(result)} 条")
    print(f"   负面 (0): {len(result) - sum(result)} 条")
    
    print("\n" + "=" * 50)
    print("完成！请上传到 Kaggle 查看成绩")
    print("=" * 50)


if __name__ == "__main__":
    main()
