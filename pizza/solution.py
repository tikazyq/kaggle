from ggplot import ggplot, aes, geom_density
import nltk
import pandas
import numpy as np

idVar = 'request_id'
titleVar = 'request_title'
textVar = 'request_text_edit_aware'
outVar = 'requester_received_pizza'

data = pandas.read_json('train.json')
test = pandas.read_json('test.json')


def flatten(lst):
    for x in lst:
        if isinstance(x, list):
            for x in flatten(x):
                yield x
        else:
            yield x


def prepare_document(x):
    x = x.lower()
    stripped = list("',.?()[]!\n\t-")
    for r in stripped:
        x = x.replace(r, ' ')
    x = [w for w in nltk.word_tokenize(x) if w not in nltk.corpus.stopwords.words('english')]
    # x = [nltk.PorterStemmer().stem(w) for w in nltk.word_tokenize(x)]
    return x


print('start generating all words')
outcomes = data[outVar].apply(lambda x: 1 if x else 0)
words = data[textVar].apply(lambda x: prepare_document(x)).tolist()
nrow = len(words)
wordbag = list(flatten(words))
worduniq = set(wordbag)

wordfreq = {key: 0 for key in worduniq}
wordscore = {key: 0 for key in worduniq}
wordprobs = {key: 0 for key in worduniq}

for w in wordbag:
    wordfreq[w] += 1
for i in range(nrow):
    for j in words[i]:
        wordscore[j] += outcomes[i]
for i in wordprobs:
    wordprobs[i] = float(wordscore[i]) / wordfreq[i]

b1 = pandas.Series([np.mean([wordprobs[w] for w in l]) if len(l) else 0 for l in words])
d1 = pandas.concat((b1, outcomes), axis=1)
d1.columns = ['prob', 'out']
g1 = ggplot(d1, aes(x='prob', color='out')) + geom_density()

cutoff = 0.3

bpreds = b1 > cutoff
print(float(sum(bpreds == outcomes)) / nrow)  # 0.922524752475

# classify test data
words = test[textVar].apply(lambda x: prepare_document(x)).tolist()
b2 = [np.mean([wordprobs.get(w) if w in wordprobs else 0 for w in l]) if len(l) else 0 for l in words]
with open('submission.csv', 'wb') as f:
    f.write(idVar + ',' + outVar + '\n')
    for i in range(len(b2)):
        row = test[idVar][i] + ',' + ('1' if b2[i] > cutoff else '0')
        print(row)
        f.write(row + '\n')

print('finished')