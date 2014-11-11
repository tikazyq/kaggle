import re
import nltk
import numpy as np


def tokenize(text):
    tokens = re.split(r'\W*', text.lower())
    tokens = [w for w in tokens if w not in nltk.corpus.stopwords.words('english') and len(w) >= 3]
    return tokens


class NaiveBayesClassifier(object):
    """
    naive bayes - document classifier

    p(ci|W) = (p(W|ci) * p(ci)) / p(W)
    where
        p(W|ci) = p(w0,w1,...wN|ci) = p(w0|ci) * p(w1|ci) * ... * p(wN|ci)
        p(ci) = number of documents for class i / total number of documents
        p(W) = p(w0,w1,...wN) = 1

    for text classification
        p(wj|ci) = probability of word wj occurring in class ci
                 = word count of wj for class ci / total word count for class ci
    """

    def __init__(self, train_matrix=None, class_labels=None, method='log'):
        self.Pcw_matrix = None
        self.Pc_vec = None
        self.class_labels = None if class_labels is None else class_labels
        self.method = None if method is None else method

    def _train(self, train_matrix, train_labels):
        nrows_train = len(train_matrix)
        ncols = len(train_matrix[0])
        class_labels = list(set(train_labels))
        nrows_classes = len(class_labels)
        # initialize class probs Pc - p(ci)
        Pc_vec = np.zeros(nrows_classes)
        # initialize class probabilities Pcw - p(wj|ci)
        Pcw_nom = np.array([[1.0] * ncols for i in range(nrows_classes)])
        Pcw_denom = np.array([2.0 for i in range(nrows_classes)])
        # iterate Pcw, Pc
        for i in range(nrows_train):
            c = class_labels.index(train_labels[i])
            Pc_vec[c] += 1
            Pcw_nom[c] += train_matrix[i]
            Pcw_denom[c] += sum(train_matrix[i])
        # compute Pcw
        # vec[np.newaxis] transform 1-D vector into 2-D, which allows for dot division
        Pcw_matrix = np.log(Pcw_nom / Pcw_denom[np.newaxis].T)
        # compute Pc
        Pc_vec = Pc_vec / Pc_vec.sum()

        # store values
        self.Pcw_matrix = Pcw_matrix
        self.Pc_vec = Pc_vec
        self.class_labels = class_labels

    def train(self, train_matrix, train_labels):
        self._train(train_matrix, train_labels)

    def calculate(self, vec):
        return (self.Pcw_matrix * vec).sum(1) + np.log(self.Pc_vec)

    def _predict(self, vec):
        res = list(self.calculate(vec))
        return self.class_labels[res.index(max(res))]

    def predict(self, vec):
        return self._predict(vec)


class NaiveBayesTextClassifier(NaiveBayesClassifier):
    def __init__(self, text_list=None):
        super(NaiveBayesTextClassifier, self).__init__()
        self.text_list = None if text_list is None else text_list
        self.vocab_list = None

    def vectorize_text(self, text):
        vocab_list = self.vocab_list
        vec = np.zeros(len(vocab_list))
        words = text if type(text) == list else tokenize(text)
        for w in words:
            if w in vocab_list:
                vec[vocab_list.index(w)] += 1
        return vec

    def vectorize_texts(self, text_list=list(), ignore_top_threshold=0, top_word_threshold=None):
        """
        convert a list of documents / texts into text matrix.
        return (text_matrix, vocab_list)
        :param text_list:
        :param ignore_top_threshold:
        :param top_word_threshold:
        :return:
        """
        word_counts = {}
        words_list = []
        for i, text in enumerate(text_list):
            words = tokenize(text)
            for w in words:
                if word_counts.get(w) is None:
                    word_counts[w] = 0
                word_counts[w] += 1
            words_list.append(words)
        # vocab list / all words set
        vocab_list = sorted([(w, word_counts[w]) for w in word_counts], key=(lambda x: x[1]), reverse=True)
        if ignore_top_threshold:
            vocab_list = vocab_list[ignore_top_threshold:]
        if top_word_threshold:
            vocab_list = vocab_list[:top_word_threshold]
        vocab_list = [x[0] for x in vocab_list]
        self.text_list = text_list
        self.vocab_list = vocab_list
        # text matrix
        matrix = []
        for i, words in enumerate(words_list):
            matrix.append(self.vectorize_text(words))
        return np.array(matrix)

    def train(self, train_text_list, train_labels, ignore_top_threshold=0, top_word_threshold=None):
        train_matrix = self.vectorize_texts(train_text_list, ignore_top_threshold=0,
                                            top_word_threshold=None)
        self._train(train_matrix, train_labels)

    def predict(self, text):
        vec = self.vectorize_text(text)
        return self._predict(vec)


class SklearnNaiveBayesTextClassifier(NaiveBayesTextClassifier):
    def __init__(self):
        super(SklearnNaiveBayesTextClassifier, self).__init__()
        from sklearn.naive_bayes import GaussianNB

        self.clf = GaussianNB()

    def _train(self, train_matrix, train_labels):
        train_matrix = train_matrix / train_matrix.sum(1)[np.newaxis].T
        self.clf.fit(train_matrix, train_labels)

    def _classify(self, vec):
        return self.clf.predict(vec)


def main():
    pass


if __name__ == '__main__':
    main()