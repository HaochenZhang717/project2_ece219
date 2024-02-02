from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np
import umap.umap_ as umap


#question 10
dataset = fetch_20newsgroups(subset='all', shuffle=True,remove=('headers','footers'))

tfidf_transformer = TfidfTransformer()
vectorizer = CountVectorizer(min_df=3, stop_words='english')
data_vector = vectorizer.fit_transform(dataset.data)
tfidf_mat = tfidf_transformer.fit_transform(data_vector).toarray()
tfidf_mat.astype(np.float32)

svd = TruncatedSVD(n_components=50)
svd_reduced = svd.fit_transform(tfidf_mat)


nmf = NMF(n_components=50, max_iter=400)
nmf_reduced = nmf.fit_transform(tfidf_mat)


kmeans_svd = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=900)
kmeans_svd.fit(svd_reduced)
print("=======================================================")
print("result of SVD+Kmeans:")
print(f"homogeneity score: {homogeneity_score(kmeans_svd.labels_, dataset.target)}")
print(f"completeness score: {completeness_score(kmeans_svd.labels_, dataset.target)}")
print(f"v_measure_score: {v_measure_score(kmeans_svd.labels_, dataset.target)}")
print(f"adjusted_rand_score: {adjusted_rand_score(kmeans_svd.labels_, dataset.target)}")
print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(kmeans_svd.labels_, dataset.target)}")
cm = confusion_matrix(dataset.target, kmeans_svd.labels_)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15,15))


kmeans_nmf = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=900)
kmeans_nmf.fit(nmf_reduced)
print("=======================================================")
print("result of NMF+Kmeans:")
print(f"homogeneity score: {homogeneity_score(kmeans_nmf.labels_, dataset.target)}")
print(f"completeness score: {completeness_score(kmeans_nmf.labels_, dataset.target)}")
print(f"v_measure_score: {v_measure_score(kmeans_nmf.labels_, dataset.target)}")
print(f"adjusted_rand_score: {adjusted_rand_score(kmeans_nmf.labels_, dataset.target)}")
print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(kmeans_nmf.labels_, dataset.target)}")
cm = confusion_matrix(dataset.target, kmeans_nmf.labels_)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15,15))

#question 11
n_components = [5,20,200]
cosine_mean_score = []
for n_comp in n_components:
    umap_cos = umap.UMAP(n_components=n_comp, metric='cosine')
    kmeans_umap = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=900)
    umap_reduced = umap_cos.fit_transform(tfidf_mat)
    kmeans_umap.fit(umap_reduced)
    print("=======================================================")
    print(f"result of UMAP+Kmeans with n_component={n_comp} and metric=cosine:")
    print(f"homogeneity score: {homogeneity_score(kmeans_umap.labels_, dataset.target)}")
    print(f"completeness score: {completeness_score(kmeans_umap.labels_, dataset.target)}")
    print(f"v_measure_score: {v_measure_score(kmeans_umap.labels_, dataset.target)}")
    print(f"adjusted_rand_score: {adjusted_rand_score(kmeans_umap.labels_, dataset.target)}")
    print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(kmeans_umap.labels_, dataset.target)}")
    cm = confusion_matrix(dataset.target, kmeans_umap.labels_)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15, 15))
    cosine_mean_score.append(
        homogeneity_score(kmeans_umap.labels_, dataset.target) \
        + completeness_score(kmeans_umap.labels_, dataset.target) \
        + v_measure_score(kmeans_umap.labels_, dataset.target) \
        + adjusted_rand_score(kmeans_umap.labels_, dataset.target) \
        + adjusted_mutual_info_score(kmeans_umap.labels_, dataset.target)
    )

euclidean_mean_score = []
for n_comp in n_components:
    umap_euc = umap.UMAP(n_components=n_comp, metric='euclidean')
    kmeans_umap = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=900)
    umap_reduced = umap_euc.fit_transform(tfidf_mat)
    kmeans_umap.fit(umap_reduced)
    print("=======================================================")
    print(f"result of UMAP+Kmeans with n_component={n_comp} and metric=euclidean:")
    print(f"homogeneity score: {homogeneity_score(kmeans_umap.labels_, dataset.target)}")
    print(f"completeness score: {completeness_score(kmeans_umap.labels_, dataset.target)}")
    print(f"v_measure_score: {v_measure_score(kmeans_umap.labels_, dataset.target)}")
    print(f"adjusted_rand_score: {adjusted_rand_score(kmeans_umap.labels_, dataset.target)}")
    print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(kmeans_umap.labels_, dataset.target)}")
    cm = confusion_matrix(dataset.target, kmeans_umap.labels_)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15, 15))
    euclidean_mean_score.append(
        homogeneity_score(kmeans_umap.labels_, dataset.target) \
        + completeness_score(kmeans_umap.labels_, dataset.target) \
        + v_measure_score(kmeans_umap.labels_, dataset.target) \
        + adjusted_rand_score(kmeans_umap.labels_, dataset.target) \
        + adjusted_mutual_info_score(kmeans_umap.labels_, dataset.target)
    )


# question 14
umap_best = umap.UMAP(n_components=n_comp_best_above, metric=best_metric_above)
umap_reduced_best = umap_best.fit_transform(tfidf_mat)
agg_ward = AgglomerativeClustering(n_clusters=20, linkage='ward')
agg_ward.fit(umap_reduced_best)
print("=======================================================")
print("result of Agglomerative Clustering with ward linkage criteria:")
print(f"homogeneity score: {homogeneity_score(agg_ward.labels_, dataset.target)}")
print(f"completeness score: {completeness_score(agg_ward.labels_, dataset.target)}")
print(f"v_measure_score: {v_measure_score(agg_ward.labels_, dataset.target)}")
print(f"adjusted_rand_score: {adjusted_rand_score(agg_ward.labels_, dataset.target)}")
print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(agg_ward.labels_, dataset.target)}")


agg_single = AgglomerativeClustering(n_clusters=20, linkage='single')
agg_single.fit(umap_reduced_best)
print("=======================================================")
print("result of Agglomerative Clustering with single linkage criteria:")
print(f"homogeneity score: {homogeneity_score(agg_single.labels_, dataset.target)}")
print(f"completeness score: {completeness_score(agg_single.labels_, dataset.target)}")
print(f"v_measure_score: {v_measure_score(agg_single.labels_, dataset.target)}")
print(f"adjusted_rand_score: {adjusted_rand_score(agg_single.labels_, dataset.target)}")
print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(agg_single.labels_, dataset.target)}")


# question 15

min_cluster_sizes = [20, 100, 200]

for min_cluster_size in min_cluster_sizes:
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    hdb.fit(umap_reduced_best)
    print("=======================================================")
    print(f"result of HDBSCAN with min_cluster_size={min_cluster_size}:")
    print(f"homogeneity score: {homogeneity_score(hdb.labels_, dataset.target)}")
    print(f"completeness score: {completeness_score(hdb.labels_, dataset.target)}")
    print(f"v_measure_score: {v_measure_score(hdb.labels_, dataset.target)}")
    print(f"adjusted_rand_score: {adjusted_rand_score(hdb.labels_, dataset.target)}")
    print(f"adjusted_mutual_info_score: {adjusted_mutual_info_score(hdb.labels_, dataset.target)}")
    cm = confusion_matrix(dataset.target, hdb.labels_)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols, yticklabels=rows, size=(15,15))


