

r = redis.redis(
    host="gusc1-assuring-dragon-30398.upstash.io",
    port=30398,
    password="7811262f644141d286a5dcb7478802bf",
    ssl=True  # Upstash typically requires SSL
)

r.set("foo", "bar")
print(r.get("foo"))
