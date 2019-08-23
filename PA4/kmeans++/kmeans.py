import numpy as np
from collections import Counter

########################################################################################
def euclidian_dist(x, A):
    A_len = np.shape(A)[0]
    x1 = np.asarray([x] * A_len)
    diff = np.subtract(x1, A)
    dist = np.linalg.norm(diff,axis=1)

    return dist

########################################################################################
# def euclidian_dist_matrix(points,centers):
def calculate_distances(centroids, x):
    all_distances = []
    for center in centroids:
        distances = euclidian_dist(center,x)
        # print(center)
        # print('===================')
        # print(x)
        # print('_______________________')
        # print(distances)
        # exit()
        all_distances.append(distances)
    return np.asarray(all_distances)

########################################################################################
def calculate_memberships(distances):
    memberships = np.argmin(distances,axis=0)

    return memberships

########################################################################################
def calculate_cost(distances):

    return np.sum(np.min(distances,axis=0)**2)

########################################################################################
def update_mean(n_clusters,memberships, x):

    new_centroids = []
    for k in range(n_clusters):
        new_centroids.append(np.mean(x[np.where(memberships==k),:][0],axis=0))

    return np.asarray(new_centroids)
########################################################################################
def get_k_means_plus_plus_center_indices_bk(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    initial_point = generator.choice(list(range(n)), 1)
    centers = [initial_point[0]]

    x_copy = np.asarray(x.copy())
    np_x = np.asarray(x)

    L = np.shape(x_copy)[0]
    x_copy[list(range(L)).remove(centers[0])]

    for k in range(n_cluster - 1):

        dists = []
        L = list(range(np.shape(np_x)[0]))

        for c in sorted(centers, reverse=True):
            L.remove(c)

        min_dist = float('inf')

        for i in L:

            dist_x = np.min(euclidian_dist(np_x[i], np_x[tuple(centers), :]))

            dists.append((i, (dist_x ** 2)))
            #  / np.sum(dist_x ** 2))
            if min_dist > dist_x:
                min_dist = dist_x

        sorted_dists = sorted(dists, key=lambda t: t[1], reverse=True)

        centers.append(sorted_dists[0][0])

    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

########################################################################################

def compute_next_center(centers,x):

    all_distances = []

    for cntr in centers:

        distances = euclidian_dist(x[cntr,:],x)

        all_distances.append(distances)

    all_distances = np.asarray(all_distances)
    min_dists = np.min(all_distances,axis=0)

    next_center = np.argmax(min_dists)

    return next_center


########################################################################################
def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    initial_point = generator.choice(list(range(n)), 1)
    centers = [initial_point[0]]


    for k in range(n_cluster - 1):

        next_center = compute_next_center(centers,x)

        centers.append(next_center)

    # raise Exception(
    #          'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers
#####################################################################################################
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

#####################################################################################################
class KMeans():
    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE

        ## Initialization Part
        J = 10**10

        centroids = x[self.centers,:]

        ## Iterative Step
        count = 0

        while count < 35:

            ## Calculate Membership
            distances = calculate_distances(centroids, x)
            memberships = calculate_memberships(distances)

            J_new = calculate_cost(distances)

            if np.abs(J-J_new) <= self.e:
                break

            J = J_new

            ## recompute means
            centroids = update_mean(self.n_cluster,memberships, x)

            count += 1

        y = memberships
        #self.max_iter = 100 if count== 35 else count


        # raise Exception(
        #      'Implement fit function in KMeans class')

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter


class KMeansClassifier():
    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        kmeans = KMeans(self.n_cluster)
        centroids, memberships, _ = kmeans.fit(x, centroid_func=centroid_func)
        centroid_labels = []

        for k in range(self.n_cluster):
            centroid_labels.append(Counter(y[np.where(memberships == k)]).most_common(1)[0][0])

        centroid_labels = np.asarray(centroid_labels)
        centroids = np.asarray(centroids)
        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement fit function in KMeansClassifier class')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        all_dists = []

        for mu in self.centroids:
            all_dists.append(euclidian_dist(mu, x))
        cluster_belongingness = np.argmin(all_dists, axis=0)
        labels = []
        for point in range(N):
            labels.append(self.centroid_labels[cluster_belongingness[point]])

        labels = np.asarray(labels)
        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #      'Implement predict function in KMeansClassifier class')

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    h, w, c = np.shape(image)

    data = image.reshape(h * w, c)

    # print(np.shape(data))

    all_dists = []

    for vector in code_vectors:
        all_dists.append(euclidian_dist(vector, data))
    memberships = np.argmin(all_dists, axis=0)

    new_data = []
    for point in memberships:
        new_data.append(code_vectors[point])

    new_im = np.reshape(new_data, (h, w, c))

    row_num = 0
    col_num = 0

    for ind, pixel in enumerate(new_data):
        # print(row_num, ' : ', col_num)
        new_im[row_num][col_num][0] = pixel[0]
        new_im[row_num][col_num][1] = pixel[1]
        new_im[row_num][col_num][2] = pixel[2]

        if col_num + 1 == w:
            col_num = 0
            row_num = row_num + 1
        else:
            col_num += 1

    # print(np.shape(new_im))

    # DONOT CHANGE CODE ABOVE THIS LINE
    # raise Exception(
    #          'Implement transform_image function')

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

