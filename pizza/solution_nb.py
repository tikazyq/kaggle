import pandas
import naive_bayes as nb

idVar = 'request_id'
titleVar = 'request_title'
textVar = 'request_text_edit_aware'
outVar = 'requester_received_pizza'

train = pandas.read_json('train.json')
test = pandas.read_json('test.json')

train_text_list = [train[titleVar][i] + ' ' + train[textVar][i] for i in range(len(train))]
test_text_list = [test[titleVar][i] + ' ' + test[textVar][i] for i in range(len(test))]

clf = nb.NaiveBayesTextClassifier()
clf.train(train_text_list, train[outVar])

with open('submission_nb.csv', 'wb') as f:
    f.write(idVar + ',' + outVar + '\n')
    for i in range(len(test)):
        row = test[idVar][i] + ',' + ('1' if clf.classify(test_text_list[i]) else '0')
        print(row)
        f.write(row + '\n')