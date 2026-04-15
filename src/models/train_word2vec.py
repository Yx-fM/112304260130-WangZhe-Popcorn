"""
Part 3: Word2Vec 词向量训练与分类

Part 3A: 训练 Word2Vec 模型
Part 3B: 使用词向量平均创建句子特征并训练分类器
"""

import pandas as pd
import numpy as np
import os
import pickle
import datetime
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import re
from bs4 import BeautifulSoup


# 数据路径
DATA_DIR = "../../data"
PROCESSED_DIR = "../../data/processed"
MODELS_DIR = "../models"
SUBMISSION_DIR = "../submissions"


def review_to_words(raw_review):
    """去除 HTML、标点，转小写"""
    markup = BeautifulSoup(raw_review, 'html.parser')
    markup_text = markup.get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", markup_text)
    words = letters_only.lower().split()
    return " ".join(words)


def make_feature_vector(words, model, num_features):
    """计算句子/评论的特征向量（词向量平均）"""
    feature_vec = np.zeros((num_features), dtype="float32")
    nwords = 0
    
    for word in words:
        if word in model.wv:
            feature_vec += model.wv[word]
            nwords += 1
    
    if nwords > 0:
        feature_vec = feature_vec / nwords
    
    return feature_vec


def get_avg_feature_vector(review, model, num_features):
    """将整条评论转为特征向量"""
    words = review_to_words(review).split()
    return make_feature_vector(words, model, num_features)


def main():
    print("=" * 60)
    print("Part 3: Word2Vec 词向量训练与分类")
    print("=" * 60)
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # ========== Part 3A: Word2Vec 训练 ==========
    print("\n【Part 3A: Word2Vec 模型训练】\n")
    
    # 加载句子数据
    print("1. 加载句子数据...")
    with open(os.path.join(PROCESSED_DIR, 'all_sentences.pkl'), 'rb') as f:
        all_sentences = pickle.load(f)
    
    print(f"   总句子数：{len(all_sentences):,}")
    
    # Word2Vec 参数
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context_window = 10
    downsampling = 1e-3
    
    print(f"\n2. 训练 Word2Vec 模型...")
    print(f"   向量维度：{num_features}")
    print(f"   最小词频：{min_word_count}")
    print(f"   上下文窗口：{context_window}")
    print(f"   句子数：{len(all_sentences):,}")
    
    model = Word2Vec(
        sentences=all_sentences,
        vector_size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        window=context_window,
        sample=downsampling,
        sg=1,
        epochs=10
    )
    
    print("   ✓ 训练完成!")
    
    # 模型统计
    vocab_size = len(model.wv.key_to_index)
    print(f"\n3. 模型统计:")
    print(f"   词汇表大小：{vocab_size:,} 词")
    
    if 'movie' in model.wv:
        print(f"\n   词向量示例：'movie' (前 20 维)\")
        print(f"   {model.wv['movie'][:20]}")
    
    # 保存模型
    model_path = os.path.join(MODELS_DIR, 'word2vec_model.bin')
    model.save(model_path)
    print(f"\n4. 模型已保存：{model_path}")
    
    # ========== Part 3B: 分类 ==========
    print("\n【Part 3B: 句子向量分类】\n")
    
    # 加载原始数据
    print("1. 加载原始数据...")
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
    
    # 创建特征向量
    print("\n2. 创建训练集特征向量...")
    train_features = np.zeros((len(train), num_features), dtype="float32")
    
    for i in range(len(train)):
        if (i+1) % 5000 == 0:
            print(f"   已处理 {i+1}/{len(train)} 条")
        train_features[i] = get_avg_feature_vector(train['review'][i], model, num_features)
    
    print("\n3. 创建测试集特征向量...")
    test_features = np.zeros((len(test), num_features), dtype="float32")
    
    for i in range(len(test)):
        if (i+1) % 5000 == 0:
            print(f"   已处理 {i+1}/{len(test)} 条")
        test_features[i] = get_avg_feature_vector(test['review'][i], model, num_features)
    
    print(f"\n   ✓ 特征矩阵:")
    print(f"     训练集：{train_features.shape}")
    print(f"     测试集：{test_features.shape}")
    
    # 训练分类器
    print("\n4. 训练分类器...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train['sentiment'],
        test_size=0.2, random_state=42
    )
    
    print(f"   训练集：{X_train.shape}, 验证集：{X_val.shape}")
    
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        verbose=0
    )
    
    forest.fit(X_train, y_train)
    
    # 验证集评估
    y_pred = forest.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f"\n   ✓ 验证集准确率：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # 全量训练并预测
    print("\n5. 全量训练并生成提交文件...")
    forest.fit(train_features, train['sentiment'])
    result = forest.predict(test_features)
    
    output = pd.DataFrame({
        'id': test['id'],
        'sentiment': result
    })
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(SUBMISSION_DIR, f"word2vec_{timestamp}.csv")
    output.to_csv(submission_path, index=False, quoting=3)
    
    print(f"   ✓ 提交文件：{submission_path}")
    print(f"   正面 (1): {sum(result)} 条")
    print(f"   负面 (0): {len(result) - sum(result)} 条")
    
    # 结果对比
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"Part 1 (BoW) 验证集准确率：~86-87%")
    print(f"Part 3 (Word2Vec) 验证集准确率：{val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"提升：{(val_accuracy - 0.87)*100:+.2f}%")
    print("=" * 60)
    print("\n✓ 完成！请上传提交文件到 Kaggle 查看成绩")
    print("=" * 60)


if __name__ == "__main__":
    main()
