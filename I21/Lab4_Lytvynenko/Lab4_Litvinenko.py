from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random

# === 1. Завантаження FastText-векторів ===
print("Завантаження FastText-векторів...")
lang1_embeddings = KeyedVectors.load_word2vec_format("cc.uk.300.bin", binary=True)
lang2_embeddings = KeyedVectors.load_word2vec_format("cc.en.300.bin", binary=True)

# === 2. Завантаження словників ===
def get_dict(file_name):
    df = pd.read_csv(file_name, delimiter=' ', header=None)
    return dict(zip(df[0], df[1]))

l1_l2_train = get_dict("l1-l2.train.txt")
l1_l2_test = get_dict("l1-l2.test.txt")

print("Розмір словника перекладів для тренування:", len(l1_l2_train))
print("Розмір словника перекладів для тестування:", len(l1_l2_test))

# === 3. Побудова матриць ===
def build_matrices(dictionary, emb1, emb2):
    X, Y = [], []
    for w1, w2 in dictionary.items():
        if w1 in emb1 and w2 in emb2:
            X.append(emb1[w1])
            Y.append(emb2[w2])
    return np.vstack(X), np.vstack(Y)

X, Y = build_matrices(l1_l2_train, lang1_embeddings, lang2_embeddings)

# === 4. Навчання трансформації ===
def gradient_descent(X, Y, lr=0.01, epochs=1000):
    R = np.random.rand(X.shape[1], X.shape[1])
    for _ in range(epochs):
        grad = -2 * X.T @ (Y - X @ R)
        R -= lr * grad
    return R

print("Навчання перетворення...")
R = gradient_descent(X, Y)

# === 5. Переклад слова ===
def translate(word, emb1, emb2, R):
    if word not in emb1:
        return None
    vec = emb1[word] @ R
    sims = {w: cosine_similarity([vec], [v])[0][0] for w, v in emb2.key_to_index.items()}
    return max(sims, key=sims.get)

# === 6. Оцінка ===
def evaluate(test_dict, emb1, emb2, R):
    correct = 0
    total = 0
    for w1, w2 in test_dict.items():
        if w1 not in emb1 or w2 not in emb2:
            continue
        translated = translate(w1, emb1, emb2, R)
        if translated == w2:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

acc = evaluate(l1_l2_test, lang1_embeddings, lang2_embeddings, R)
print(f"Точність перекладу: {acc:.2%}")

# === 7. Завантаження твітів ===
def load_tweets(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

tweets = load_tweets("tweets.txt")

# === 8. Векторизація твіту ===
def tweet_to_vector(tweet, embeddings):
    tokens = tweet.lower().split()
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    return np.sum(vectors, axis=0) if vectors else np.zeros(embeddings.vector_size)

tweet_vectors = [tweet_to_vector(t, lang1_embeddings) for t in tweets]

# === 9–10. LSH пошук ===
def create_lsh_planes(dim, num_planes):
    return np.random.randn(num_planes, dim)

def hash_vector(vec, planes):
    return tuple((vec @ plane > 0).astype(int) for plane in planes)

def build_lsh_index(vectors, planes):
    index = {}
    for i, vec in enumerate(vectors):
        h = hash_vector(vec, planes)
        index.setdefault(h, []).append(i)
    return index

def query_lsh(query_vec, planes, index, tweet_vectors, top_k=5):
    h = hash_vector(query_vec, planes)
    candidates = index.get(h, [])
    similarities = [(i, cosine_similarity([query_vec], [tweet_vectors[i]])[0][0]) for i in candidates]
    return sorted(similarities, key=lambda x: -x[1])[:top_k]

print("Побудова LSH індексу...")
planes = create_lsh_planes(lang1_embeddings.vector_size, 10)
lsh_index = build_lsh_index(tweet_vectors, planes)

# === 11. Тестовий запит ===
idx = random.randint(0, len(tweets) - 1)
query = tweet_vectors[idx]
results = query_lsh(query, planes, lsh_index, tweet_vectors)

print("\nОригінальний твіт:")
print(tweets[idx])
print("\nСхожі твіти:")
for i, score in results:
    print(f"[{score:.4f}] {tweets[i]}")
