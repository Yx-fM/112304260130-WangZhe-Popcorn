"""
Part 2: 句子分词与 Word2Vec 数据准备

使用 NLTK 将评论拆分为句子，为 Word2Vec 训练做准备
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk


def review_to_words(raw_review):
    """去除 HTML、标点，转小写"""
    markup = BeautifulSoup(raw_review, 'html.parser')
    markup_text = markup.get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", markup_text)
    words = letters_only.lower().split()
    return " ".join(words)


def make_sentences_list(reviews, max_sentences=100):
    """
    将评论列表转换为句子列表（每个句子是单词列表）
    """
    sentences = []
    for i, review in enumerate(reviews):
        if (i+1) % 5000 == 0:
            print(f"   已处理 {i+1}/{len(reviews)} 条")
        
        # 清洗
        clean_review = review_to_words(review)
        
        # 句子分词
        review_sentences = sent_tokenize(clean_review)
        
        # 限制句子数量并分词
        for sent in review_sentences[:max_sentences]:
            words = word_tokenize(sent)
            if len(words) > 0:
                sentences.append(words)
    
    return sentences


def main():
    # 数据路径
    DATA_DIR = "../../data"
    PROCESSED_DIR = "../../data/processed"
    
    # 确保目录存在
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Part 2: 句子分词与 Word2Vec 数据准备")
    print("=" * 50)
    
    # 下载 NLTK 数据
    print("\n1. 准备 NLTK...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("   下载 punkt 分词器...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
    print("   ✓ NLTK 就绪")
    
    # 加载数据
    print("\n2. 加载数据...")
    train = pd.read_csv(
        os.path.join(DATA_DIR, "labeledTrainData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    unlabeled = pd.read_csv(
        os.path.join(DATA_DIR, "unlabeledTrainData.tsv"),
        header=0, delimiter="\t", quoting=3
    )
    print(f"   训练集：{train.shape}")
    print(f"   无标签数据：{unlabeled.shape}")
    
    # 处理训练集
    print("\n3. 处理训练集...")
    train_sentences = make_sentences_list(train['review'].tolist())
    print(f"   ✓ 训练集句子数：{len(train_sentences)}")
    
    # 处理无标签数据
    print("\n4. 处理无标签数据 (用于 Word2Vec 训练)...")
    unlabeled_sentences = make_sentences_list(unlabeled['review'].tolist())
    print(f"   ✓ 无标签数据句子数：{len(unlabeled_sentences)}")
    
    # 合并数据
    all_sentences = train_sentences + unlabeled_sentences
    print(f"\n   总句子数：{len(all_sentences)}")
    
    # 统计信息
    print("\n5. 数据统计...")
    sentence_lengths = [len(sent) for sent in all_sentences]
    print(f"   句子长度：min={min(sentence_lengths)}, max={max(sentence_lengths)}, avg={np.mean(sentence_lengths):.1f}")
    
    all_words = [word for sent in all_sentences for word in sent]
    unique_words = set(all_words)
    print(f"   总词数：{len(all_words)}")
    print(f"   不重复词数：{len(unique_words)}")
    
    # 词频统计
    from collections import Counter
    word_freq = Counter(all_words)
    print(f"\n   最常见单词:")
    for word, count in word_freq.most_common(10):
        print(f"     {word}: {count}")
    
    # 保存数据
    print("\n6. 保存处理后的数据...")
    with open(os.path.join(PROCESSED_DIR, 'train_sentences.pkl'), 'wb') as f:
        pickle.dump(train_sentences, f)
    with open(os.path.join(PROCESSED_DIR, 'all_sentences.pkl'), 'wb') as f:
        pickle.dump(all_sentences, f)
    print(f"   ✓ 保存到 {PROCESSED_DIR}/")
    
    print("\n" + "=" * 50)
    print("完成！数据已保存，可用于 Part 3 Word2Vec 训练")
    print("=" * 50)


if __name__ == "__main__":
    main()
