import tiktoken

text = "Привет, мир! Как дела? 112637182371"

# Кодировка для модели
encoding = tiktoken.get_encoding("cl100k_base")
token_ids = encoding.encode(text)

print("Исходный текст:")
print(text)

print("\nID токенов:")
print(token_ids)

print("\nТокены по отдельности:")
for token_id in token_ids:
    token_bytes = encoding.decode_single_token_bytes(token_id)
    print(f"{token_id}: -> {token_bytes.decode('utf-8', errors='replace')}")