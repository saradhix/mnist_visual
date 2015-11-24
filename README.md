Code is given under MIT Licence. Feel free to use it whatever way you want.
It is difficult to visualise data in higher dimensions. We can plot data in 2 or atmost 3 dimensions to 'understand' how the data looks like. Beyond 3 dimensions, the plots are difficult to visualise and understand. To help in the visualisation of the structure of data, we have to reduce the dimensions of the data.

In this post, we are going to look at the similarity of the handwritten digits, why some digits are always confused for something else(see the title picture), like 4 and 9 are confused frequently. Similary 1 and 7 are confused, 3 and 8 are confused.

Consider a d dimensional data set(where d>3). We can plot the data using 2 dimensions in d(d-1)/2 ways. If we have a dataset of 10 dimensions, then the possible number of drawing a 2D plots, by just using 2 dimensions can be done in 45 ways. If we take any random combination, then the plot is called a random projection In a random projection, it is likely that the more interesting structure withing the data will be lost.


Let us consider the standard MNIST dataset for our experiments. MNIST stands for Mixed National Institute of Standards and Technology. The MNIST database is a large database of handwritten digits that is commonly used for image processing and image recognition. The database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.

Lets examine the shape of the dataset. Shape of the dataset means, the number of samples and the dimensionality of the samples.


mnist = fetch_mldata("MNIST Original")
X, y = mnist.data / 255.0, mnist.target
n_train_samples = 1000
print X.shape

70000, 784

Yes, MNIST has 70000 samples(60000 train + 10000 test). Each sample has 784 dimensions. Infact, each sample is a 28x28 array of integers, each integer ranging from 0-255, showing the intensity of the blackness of the image at that location.

I want to show how we can differentiate between the frequently confused numbers by Machine Learning programs by using some manifold learning techniques. Lets first understand why the image classifier gets confused. This image presents the most frequently confused samples.

It can be observed that the sample 119, the classifier incorrectly classifies the image as 9, when the true digit is a 4. The classifier incorrectly classifies the image sample 322 as 7, when the digit was a 2.

A number of supervised and unsupervised linear dimensionality reduction frameworks have been designed, such as Principal Component Analysis (PCA), Independent Component Analysis, Linear Discriminant Analysis, and others. Lets look at some of these.

PCA
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. The number of principal components is less than or equal to the number of original variables.

Let us plot the MNIST digits on a 2 dimensional plane with 2 most important dimensions. 

Observe that there is very much less linear separability, in the sense that the 7s and 1s are intermixed and difficult to separate. The zeros, sixes, fives and twos at the center are almost close to each other and are not easily separable. 

Lets us plot the PCA by considering only the digits from the confused pair 4 and 9. The below plot shows the scatter plot on the two most important features extracted through PCA for those samples whose labels are 4 or 9.

In the above plot 4s are represented in Red and 9s are represented in Green. We can observe that we can not draw a straight line(or even a curve) which can separate the two classes. What this means is that the values of the most important dimensions for a 4 and 9 are almost overlapping, and it is difficult to separate by just seeing these two features alone. They might be separable in higher dimensions but we are not going to delve into higher dimensions. We can see that the extent of Red points have mixed with the Green points and they are existing close to each other, which makes separation of the classes with just these two dimensions difficult.

Lets us plot the PCA by considering only the digits from another frequently confused pair 1 and 7. The below plot shows the scatter plot on the two most important features extracted through PCA for those samples whose labels are 1 or 7.

With 1 and 7, we can see a higher degree of separability, though some 1s are close to the 7s, i.e some Red points are present in between the Green cluster. Similarly, some Green points are engulfed by the Red cluster.  If we compare and contrast with the PCA of 4 and 9, it can be observed that 1 and 7 are more separable than 4 and 9.

PCA is a linear transformation of the dimensions, and often miss important non-linear structure in the data. Manifold Learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data. Though supervised variants exist, the typical manifold learning problem is unsupervised: it learns the high-dimensional structure of the data from the data itself, without the use of predetermined classifications.

Some of these manifold learning techniques are

ISOMAP - Isometric Mapping
LLE - Local Linear Embedding
MDS - Multidimensional Scaling
TSNE - t-distributed Stochastic Neighbor Embedding
ISOMAP
Isomap is the earliest approach to manifold learning. Isomap stands for Isometric Mapping. Isomap seeks a lower dimensional embedding which maintains geodesic distances between all points. 
The Isomap algorithm comprises of three stages


1. Nearest Neighbour search. In this stage, a predetermined number of closest neighbours are identified for every sample. A graph is created with all points as node and an edge exists between two nodes x and y if x is a k-nearest neighbour of y. It is obvious that this is an undirected graph as distance betwen node x and node y is same as the distance between node y and node x.

2. Shortest-path graph search. In this phase, the shortest path between every pair of samples can be found using Floyd-Warshal's all pair shortest distance algorithm. A distance matrix of dimensions n X n is creatted, where n is the number of samples in the set. The entry for ith row and jth column in this distance matrix represents the distance between the sample i and sample j. Two points are to be noted - This matrix is symmetric(Why ?), and the values at principal diagonal is zero(Can you find out why ?), i.e a[i][i]=0 for 0<i<n, where n is the number of samples.

3. Partial Eigenvalue decomposition. The embedding is encoded in the eigenvectors corresponding to the d largest eigenvalues of the N X N isomap kernel

Let us see how the Isomap of the MNIST database for all the digits looks like.
The Isomap for MNIST shows that the digits are better separable than the vanilla PCA. Let us plot the Isomap for the frequently confused pairs (4,9) and (1,7).

Although not fully linearly separable, 4-9 seem to be having better separability when compare with PCA.

Similarly, Isomap for 1 and 7 shows a similar degree of separability like that of with PCA.

LLE

Local linear Embedding seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods. It can be thought of as a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding.
Similar to Isomap, LLE algorithm consists of three stages
1. Nearest Neighbour Search. This is exactly similar to the Nearest Neighbour Search in Isomap.
2. Weight Matrix Construction. This involves construction of a LLE weight matrix.
3. Partial Eigenvalue Decomposition. This is exactly similar to the s3rd stage in Isomap.
Let us see how the LLE of the MNIST database looks like.
We can see that there are some local clusters formed. We can observe that with LLE, the 4s, 7s and 9s have migrated to a cluster at the extreme right. This is because structurally the digits are very similar. We can also observe that the digits 2 and 8 have formed a cluster at the center of the figure because the structure of 2 and 8 are close. There are some clusters where 2 and 5 are together, bottom center. Lets see the LLE for the frequently confused digits.

Observe that when compare to PCA or Isomap, LLE for digits 4 and 9 show better separability.

Similarly, LLE plot for 1 and 7 shows very high degree of separability, with a very few 1s migrating to the 7 cluster.

MDS
Multidimensional scaling (MDS) is a means of visualizing the level of similarity of individual cases of a dataset. It refers to a set of related ordination techniques used in information visualization, in particular to display the information contained in a distance matrix. An MDS algorithm aims to place each object in N-dimensional space such that the between-object distances are preserved as well as possible. Each object is then assigned coordinates in each of the N dimensions. The number of dimensions of an MDS plot N can exceed 2 and is specified a priori. Choosing N=2 optimizes the object locations for a two-dimensional scatterplot.
Multidimensional scaling (MDS) seeks a low-dimensional representation of the data in which the distances respect well the distances in the original high-dimensional space.
In general, is a technique used for analyzing similarity or dissimilarity data. MDS attempts to model similarity or dissimilarity data as distances in a geometric spaces. The data can be ratings of similarity between objects, interaction frequencies of molecules, or trade indices between countries.
Note that the purpose of the MDS is to find a low-dimensional representation of the data (here 2D) in which the distances respect well the distances in the original high-dimensional space, unlike other manifold-learning algorithms, it does not seeks an isotropic representation of the data in the low-dimensional space.

Let us see how the MDS of the MNIST database looks like.
MDS for frequently confused digits 4,7 below.

MDS for frequently confused pair 1,7 below



TSNE t-distributed Stochastic Neighbor Embedding
t-SNE (TSNE) converts affinities of data points to probabilities. The affinities in the original space are represented by Gaussian joint probabilities and the affinities in the embedded space are represented by Student’s t-distributions. This allows t-SNE to be particularly sensitive to local structure and has a few other advantages over existing techniques:
Revealing the structure at many scales on a single map
Revealing data that lie in multiple, different, manifolds or clusters
Reducing the tendency to crowd points together at the center
While Isomap, LLE and variants are best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples as highlighted on the S-curve example. This ability to group samples based on the local structure might be beneficial to visually disentangle a dataset that comprises several manifolds at once as is the case in the digits dataset.
The Kullback-Leibler (KL) divergence of the joint probabilities in the original space and the embedded space will be minimized by gradient descent. Note that the KL divergence is not convex, i.e. multiple restarts with different initializations will end up in local minima of the KL divergence. Hence, it is sometimes useful to try different seeds and select the embedding with the lowest KL divergence.
The disadvantages to using t-SNE are roughly:

t-SNE is computationally expensive, and can take several hours on million-sample datasets where PCA will finish in seconds or minutes.
The Barnes-Hut t-SNE method is limited to two or three dimensional embeddings.
The algorithm is stochastic and multiple restarts with different seeds can yield different embeddings. However, it is perfectly legitimate to pick the the embedding with the least error.
Global structure is not explicitly preserved. This is problem is mitigated by initializing points with PCA (using init=’pca’).
Lets see tSNE plot for all MNIST digits.

This looks better, in a way that there is a separate cluster for every digit. In other words, all samples with the same values are clustered together and there is no mixing of the clusters. Clearly t-SNE is a superior way to visualise, reduce dimensionality for samples which exhibit manifold structure. Observe that the frequently confused pair 4 and 9 are close to each other. Also note that another frequently confused pair 1 and 7 are considerably far from each other. We can also observer that there are two clusters for the digit 2 in yellow at the top.

Lets see how the frequently confused pair 4 and 9 are depicted in the t-SNE plot.

We can observe that there is a very high degree of separation, ie. it is trivial to draw a straight line which separate the two classes, with very few non compliant samples. Lets see how the frequently confused pair 1 and 7 are depicted in the t-SNE plot.

In this case too, there is a high degree of separability between the two classes.

As a bonus, lets see how t-SNE performs with another frequently confused pair 3 and 8.

We see that there is a high degree of separability betwen the two classes, with very few samples dislocated from their respective clusters. We can see that there are few Red dots in the green cluster and few Green dots in the Red cluster.

We have visualised various manifold learning techniques like Isomap, LLE, MDS and TSNE on the MNIST dataset. We have compared the degree of separability with various manifold learning techniques on 2D plane. We can extend the same techniques to visualise the manifold on a 3D plane. We can do similar manifold learning on a similar dataset called USPS. Experiments on 3D visualisation using manifold learning and experiments on USPS dataset are left as an exercise to the interested readers.

All the code is available for public use at http://github.com/saradhix/mnist_visual

Note: Since we do random shuffles of the training set and plot the first N samples, the results you get might vary slightly.

References:
Manifold Learning - http://scikit-learn.org/stable/modules/manifold.html
Comparison of Manifold Learning Methods - http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
MNIST Database - http://yann.lecun.com/exdb/mnist/
t-SNE in scikit - http://alexanderfabisch.github.io/t-sne-in-scikit-learn.html
Handwritten Digit Classification - https://people.cs.umass.edu/~smaji/projects/digits/index.html

About the author: Vijayasaradhi Indurthi is currently pursuing his Masters at IIIT Hyderabad. His interests include Data Mining, Machine Learning and building massively scalable products.

