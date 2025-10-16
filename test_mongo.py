from pymongo import MongoClient

client = MongoClient("mongodb://127.0.0.1:27017/")
db = client.get_database("mindmate")
db.test.insert_one({"ok": True})
print("✅ MongoDB connected and working!")
