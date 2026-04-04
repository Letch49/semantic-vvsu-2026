import math
import tiktoken


text = "Я люблю"

# токенизация
enc = tiktoken.get_encoding("cl100k_base")
token_ids = enc.encode(text)

print("Исходный текст:")
print(text)
print()

print("Token IDs:")
print(token_ids)
print()

print("Токены:")
for tid in token_ids:
    piece = enc.decode([tid])
    print(f"{tid} -> {repr(piece)}")

# псевдо генерация токенов
def token_to_vector(token_id):
    x = (token_id % 10) / 10.0
    y = ((token_id // 10) % 10) / 10.0
    return [x, y]

vectors = [token_to_vector(tid) for tid in token_ids]

print("\nВекторы токенов:")
for tid, vec in zip(token_ids, vectors):
    print(f"{tid} -> {vec}")

# общий вектор контекста (грубый пример)
context = [
    sum(v[0] for v in vectors) / len(vectors),
    sum(v[1] for v in vectors) / len(vectors),
]

print("\nВектор контекста:")
print(context)

# кандидаты токенов
candidate_texts = ["кошек", "собак", "пиццу", "работать", "!"]
candidate_ids = [enc.encode(c)[0] for c in candidate_texts]

print("\nКандидаты:")
for c, cid in zip(candidate_texts, candidate_ids):
    print(f"{repr(c)} -> {cid}")

# сравниваем контекст с каждым кандидатом через скалярное произведение
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]

scores = []
for cid in candidate_ids:
    vec = token_to_vector(cid)
    score = dot(context, vec)
    scores.append(score)

# -----------------------------
# 7. Softmax -> вероятности
# -----------------------------
def softmax(values):
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    s = sum(exps)
    return [e / s for e in exps]

probs = softmax(scores)

print("\nВероятности следующего токена:")
for c, cid, score, prob in zip(candidate_texts, candidate_ids, scores, probs):
    print(f"{repr(c):>12} | id={cid:<6} | score={score:.3f} | prob={prob:.3f}")

best_index = probs.index(max(probs))
print("\nСамый вероятный следующий токен:")
print(repr(candidate_texts[best_index]))