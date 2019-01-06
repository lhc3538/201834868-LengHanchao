import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def load_data():
    data_path = "./Tweets.txt"
    corpus, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line.strip())
            corpus.append(j['text'])
            labels.append(j['cluster'])
    return corpus, labels


def my_kmeans(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    kmeans = KMeans(n_clusters=110, max_iter=60, n_init=10, init='k-means++')
    result_kmeans = kmeans.fit_predict(x.toarray())
    print('K-means NMI:', normalized_mutual_info_score(result_kmeans, labels))


def my_affinity_propagation(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    affinity_propagation = AffinityPropagation(damping=.55, max_iter=300, convergence_iter=15, copy=False)
    result_affinity_propagation = affinity_propagation.fit_predict(x.toarray())
    print('AffinityPropagation NMI:', normalized_mutual_info_score(result_affinity_propagation, labels))


def my_mean_shift(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    mean_shift = MeanShift(bandwidth=0.66, bin_seeding=True)
    result_mean_shift = mean_shift.fit_predict(x.toarray())
    print('MeanShift NMI:', normalized_mutual_info_score(result_mean_shift, labels))


def my_spectral_clustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    spectral_clustering = SpectralClustering(n_clusters=110, n_init=10)
    result_spectral_clustering = spectral_clustering.fit_predict(x.toarray())
    print('SpectralClustering NMI:', normalized_mutual_info_score(result_spectral_clustering, labels))


def my_agglomerative_clustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=110)
    result_agglomerative_clustering = agglomerative_clustering.fit_predict(x.toarray())
    print('AgglomerativeClustering NMI:', normalized_mutual_info_score(result_agglomerative_clustering, labels))


def my_dbscan(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)
    dbscan = DBSCAN(eps=0.5, min_samples=1, leaf_size=32)
    result_dbscan = dbscan.fit_predict(x.toarray())
    print('DBSCAN NMI:', normalized_mutual_info_score(result_dbscan, labels))
    # 如果min_samples 设置为默认的5，则accuracy是0.1080121348508573


def my_gaussian_mixture(corpus, labels):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)

    gaussian_mixture = GaussianMixture(n_components=110)
    result_gaussian_mixture = gaussian_mixture.fit_predict(x.toarray())
    print('GaussianMixture NMI:', normalized_mutual_info_score(result_gaussian_mixture, labels))


if __name__ == '__main__':
    corpus, labels = load_data()
    print('num_cluster:', max(labels))
    my_kmeans(corpus, labels)
    my_affinity_propagation(corpus, labels)
    my_mean_shift(corpus, labels)
    my_spectral_clustering(corpus, labels)
    my_agglomerative_clustering(corpus, labels)
    my_dbscan(corpus, labels)
    my_gaussian_mixture(corpus, labels)

# num_cluster: 110
# K-means NMI: 0.7916051205577013
# AffinityPropagation NMI: 0.7834777200368183
# MeanShift NMI: 0.7468492000608157
# SpectralClustering NMI: 0.6716412603878753
# AgglomerativeClustering NMI: 0.7758740356993199
# DBSCAN NMI: 0.7009526046894612
# GaussianMixture NMI: 0.7859487200756089