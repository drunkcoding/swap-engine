import json
a = {"int": 1234567}
a[23] = 23
a["t"] = True

with open("test.json", "w") as f:
    json.dump(a, f)

with open("test.json", "r") as f:
    b = json.load(f)

print(b)
print(type(b["t"]))