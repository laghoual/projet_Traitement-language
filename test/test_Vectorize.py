from parsers.Parse import EnglishPosParser
from vect.Vectorize import Vectorizer

parser = EnglishPosParser()
documents = parser.read_file("test_file")
vectorizer = Vectorizer("../glove.6B.50d.w2v.txt")
features, shapes = vectorizer.encode_features(documents)
labels = vectorizer.encode_annotations(documents)
assert (features.size == shapes.size == labels.size == 4)
