from typing import List
from re import fullmatch, match


class Interval:
    """class pour  representer a range continu des entier"""

    def __init__(self, start: int, end: int):
        """
        :param start: start de notre intervalle
        :param end: first integer nn'est inclus dans l intervalle
        """
        self.start = int(start)
        self.end = int(end)
        if self.start > self.end:
            raise ValueError('Start "{}" must not be greater than end "{}"'.format(self.start, self.end))
        if self.start < 0:
            raise ValueError('Start "{}" must not be negative'.format(self.start))

    def __len__(self):
        """ Retourne end-strat """
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __ne__(self, other):
        return self.start != other.start or self.end != other.end

    def __lt__(self, other):
        return (self.start, -len(self)) < (other.start, -len(other))

    def __le__(self, other):
        return (self.start, -len(self)) <= (other.start, -len(other))

    def __gt__(self, other):
        return (self.start, -len(self)) > (other.start, -len(other))

    def __ge__(self, other):
        return (self.start, -len(self)) >= (other.start, -len(other))

    def __hash__(self):
        return hash(tuple(v for k, v in sorted(self.__dict__.items())))

    def __contains__(self, item: int):
        """ Return self.start <= item < self.end """
        return self.start <= item < self.end

    def __repr__(self):
        return 'Interval[{}, {}]'.format(self.start, self.end)

    def __str__(self):
        return repr(self)

    def intersection(self, other) -> 'Interval':
        """ Retourne  the l'intersection d'intervalle to self and other """
        a, b = sorted((self, other))
        if a.end <= b.start:
            return Interval(self.start, self.start)
        return Interval(b.start, min(a.end, b.end))

    def overlaps(self, other) -> bool:
        """ Return True if there exists an interval common to self and other """
        a, b = sorted((self, other))
        return a.end > b.start

    def shift(self, i: int):
        self.start += i
        self.end += i


class Token(Interval):
    """ A Interval representing word like units  """

    def __init__(self, document, start: int, end: int, shape: int, text: str,label: str=None):
        """

        :param document: the document qui contient  le Token
        :param start: le debut de tocken in document text
        :param end: la fin de  Token in document text
        :param pos: part of speach of the Token
        :param shape: integer label describing the shape of the Token
        :param text: this is the text representation of Token
        """

        Interval.__init__(self, start, end)
        self._doc = document
        self._label = label
        self._shape = shape
        self._text = text


    @property
    def text(self):
        return self._text

    @property
    def pos(self):
        return self._pos

    @property
    def shape(self):
        return self._shape

    @property
    def label(self):
        return self._label

    def __getitem__(self, item):
        return self._text[item]

    def __repr__(self):
        return 'Token({}, {}, {})'.format(self.text, self.start, self.end)


class Sentence(Interval):
    """ Interval correspondant  to a Sentence"""

    def __init__(self, document, start: int, end: int):
        Interval.__init__(self, start, end)
        self._doc = document

    def __repr__(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        """ retourne la liste des tokens contenu dans la phrase"""
        return [token for token in self._doc.tokens if token.overlaps(self)]


class Document:
    @classmethod
    def create_from_vectors(cls, words: List[str], sentences: List[Interval]=None, labels: List[str]=None):
        doc = Document()
        text = []
        offset = 0
        doc.sentences = []
        for sentence in sentences:
            text.append(' '.join(words[sentence.start:sentence.end + 1]) + ' ')
            doc.sentences.append(Sentence(doc, offset, offset + len(text[-1])))
            offset += len(text[-1])
        doc.text = ''.join(text)

        offset = 0
        doc.tokens = []
        for word, label in zip(words, labels):
            pos = doc.text.find(word, offset)
            if pos >= 0:
                offset = pos + len(word)
                doc.tokens.append(Token(doc, pos, offset, get_shape_category(word), word, label))
        return doc


def get_shape_category_simple(word):
    if word.islower():
        return 'ALL-LOWER'
    elif word.isupper():
        return 'ALL-UPPER'
    elif fullmatch('[A-Z][a-z]+', word):
        return 'FIRST-UPPER'
    else:
        return 'MISC'


def get_shape_category(token):
    if match('^[\n]+$', token):  # IS LINE BREAK
        return 'NL'
    if any(char.isdigit() for char in token) and match('^[0-9.,]+$', token):  # IS NUMBER (E.G., 2, 2.000)
        return 'NUMBER'
    if fullmatch('[^A-Za-z0-9\t\n ]+', token):  # IS SPECIAL CHARS (E.G., $, #, ., *)
        return 'SPECIAL'
    if fullmatch('^[A-Z\-.]+$', token):  # IS UPPERCASE (E.G., AGREEMENT, INC.)
        return 'ALL-CAPS'
    if fullmatch('^[A-Z][a-z\-.]+$', token):  # FIRST LETTER UPPERCASE (E.G. This, Agreement)
        return '1ST-CAP'
    if fullmatch('^[a-z\-.]+$', token):  # IS LOWERCASE (E.G., may, third-party)
        return 'LOWER'
    if not token.isupper() and not token.islower():  # WEIRD CASE (E.G., 3RD, E2, iPhone)
        return 'MISC'
    return 'MISC'
