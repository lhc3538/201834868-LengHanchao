from sklearn import datasets, metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

dir_20newsgroups = "/home/leon/Disk/dataset/Downloads/20Newsgroups/20news-18828"
use_my_knn = False


def calculate_result(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='micro')
    m_recall = metrics.recall_score(actual, pred, average='micro')
    print('predict info:')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred, average='micro')))


# Load 20newsgroups
data_20newsgroups = datasets.load_files(dir_20newsgroups)

# 80% train_set and 20% test_set
doc_terms_train, doc_terms_test, doc_class_train, doc_class_test = \
    model_selection.train_test_split(data_20newsgroups.data, data_20newsgroups.target, test_size=0.2)

# Extract features
vectorizer_train = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english',
                                   decode_error='ignore', max_features=3000)
fea_train = vectorizer_train.fit_transform(doc_terms_train)
vectorizer_test = TfidfVectorizer(vocabulary=vectorizer_train.vocabulary_, decode_error='ignore',
                                  max_features=3000)
fea_test = vectorizer_test.fit_transform(doc_terms_test)

# Print features
feature_names_train = vectorizer_train.get_feature_names()
print(feature_names_train[-10:])
feature_names_test = vectorizer_test.get_feature_names()
print(feature_names_test[-10:])
print(fea_train.shape)
print(fea_train.nnz / float(fea_train.shape[0]))
print(fea_test.shape)
print(fea_test.nnz / float(fea_test.shape[0]))

mnb = MultinomialNB()
mnb.fit(fea_train, doc_class_train)
pred = mnb.predict(fea_test)
calculate_result(doc_class_test, pred)