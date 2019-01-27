from parsers.Parse import EnglishPosParser

parser = EnglishPosParser()
documents = parser.read_file("test_file")
assert (len(documents) == 1)
assert (len(documents[0].sentences)== 4)
assert (len(documents[0].tokens)== 43)
