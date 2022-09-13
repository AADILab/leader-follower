import pickle

pickle.dump(("Hello"), open("test.pkl", "wb"))

h = pickle.load(open("test.pkl", "rb"))
print(h)
