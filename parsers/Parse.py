from Token.Tokenize import Interval, Document


class Parser(object):
    """Classe parente pour tous les parsers  """
    def create(self):
        return self

    def read_file(self, filename: str) -> Document:
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return self.read(content)


class EnglishPosParser(Parser):
    def read(self, content: str) -> [Document]:
        """Reads the content of a NER/POS data file and returns one document instance per document it finds."""
        documents = []

        # Split le  text in documents en utilsant string '-DOCSTART- -X- O O' and loop over it
        docs = content.split("-DOCSTART- -X- O O")
        docs.pop(0)
        for doc in docs:
            sentences_list = []
            words_list = []
            tags_list = []
            sentence_start = 0
            # Slit lines and loop over
            lines = doc.split('\n\n')
            lines.pop(0)
            lines.pop()
            for line in lines:
                for vector in line.split('\n'):
                    # creer les  vectors of tokens and labels (column 2) and at the '\n\n' make a sentence.
                    word_with_tags = vector.split()
                    words_list.append(word_with_tags[0])
                    tags_list.append(word_with_tags[1])
                sentences_list.append(Interval(sentence_start, len(words_list) - 1))
                sentence_start = len(words_list)
        #     Create a Document object and append ajoute it to list of documents
            documents.append(Document.create_from_vectors(words_list, sentences_list, tags_list))

        return documents
