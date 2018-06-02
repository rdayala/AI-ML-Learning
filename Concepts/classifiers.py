import numpy as np

from sklearn import linear_model

import warnings
warnings.filterwarnings("ignore")

def find_eigenvalues_and_eigenvectors_simply(A):
    # Your code here
    L = 1 / len(A.T) * np.dot(A, A.T)
    e, u = np.linalg.eig(L)
    w = e
    v = np.dot(A.T, u)
    return w, v
    
def predict_and_find_accuracy(algorithm, train_features, train_labels, given_features, given_labels, k=1, optimized=True):

    # All data inputs must be np arrays, all labels must be integers

    # algorithm = {"kNN", "NB", "linear"}

    predicted_labels = predict(algorithm=algorithm,
                               train_features=train_features,
                               train_labels=train_labels,
                               given_features=given_features,
                               k=k,
                               optimized=optimized)

    return np.mean(predicted_labels == given_labels)


def predict(algorithm, train_features, train_labels, given_features, k=1, optimized=True):

    # All data inputs must be np arrays, all labels must be integers

    # TRAINING

    if algorithm == 'NB':
        NB_clf = naive_bayes_classifier(train_features, train_labels)

    if algorithm == 'linear':
        multiclass_linear_clf = multiclass_linear_classifier(train_features, train_labels)

    # INFERENCE

    if algorithm == 'kNN':
        predicted_labels = kNN_classify(k, train_features, train_labels, given_features)

    if algorithm == 'NB':
        predicted_labels = naive_bayes_classify(NB_clf, given_features, optimized=optimized)

    if algorithm == 'linear':
        predicted_labels = multiclass_linear_classify(multiclass_linear_clf, given_features)

    return predicted_labels


def dist(train_features, given_feature):
    return np.sqrt(np.sum((train_features - given_feature)**2, axis=1))


def kNN(k, train_features, train_labels, test_sample_features):
    # Compute distance
    tiled_test_sample_features = np.tile(test_sample_features, (len(train_features), 1))
    distances = dist(train_features, tiled_test_sample_features)
    sort_neighbors = np.argsort(distances)
    return np.concatenate((distances[sort_neighbors][:k].reshape(-1, 1), train_labels[sort_neighbors][:k].reshape(-1, 1)), axis=1)


def kNN_classify(k, train_features, train_labels, test_features):

    if len(test_features.shape) == 1:
        test_features = np.reshape(test_features, (1, len(test_features)))

    predicted_labels = []

    for t, test_sample_features in enumerate(test_features):
        # print("Progress: {0:0.04f}".format((t+1)/len(test_features)), end="\r")

        nn_labels = []
        for nn in kNN(k, train_features, train_labels, test_sample_features):
            nn_labels.append(nn[-1])

        predicted_labels.append(np.argmax(np.bincount(nn_labels)))

    # print("")

    return np.array(predicted_labels)


def naive_bayes_classifier(train_features, train_labels):

    classes, counts = np.unique(train_labels, return_counts=True)
    number_of_classes = len(classes)
    number_of_features = train_features.shape[1]

    # Prior
    prior = []
    for c, class_label in enumerate(classes):
        prior.append(counts[c]/len(train_labels))

    # Calculate the mean and variance per feature dimension here 
    # from the training set from samples belonging to each class label.
    means = np.zeros((number_of_features, number_of_classes)) # every feature, for each class
    std_dev = np.zeros((number_of_features, number_of_classes)) # every feature, for each class
    # For each class
    for c, class_label in enumerate(classes): # selecting a class 'y'
        class_rows = train_features[np.where(train_labels == class_label)[0], :]    # get all samples belonging to 'class_label'
        # For each feature
        for f in range(number_of_features):
            means[f, c] = np.mean(class_rows[:, f])
            std_dev[f, c] = np.std(class_rows[:, f])

    NB_classifier = {}
    NB_classifier['prior'] = prior
    NB_classifier['means'] = means
    NB_classifier['std_dev'] = std_dev

    return NB_classifier


def gaussian(x, m, v):
    return np.sqrt(1.0 / 2*np.pi) / v * np.exp(-0.5*(((x - m)/v)**2) )


def get_likelihood_optimized(features, means, std_dev):

    number_of_features, number_of_classes = means.shape

    means = np.tile(means.reshape(1, means.shape[0], means.shape[1]), (len(features), 1, 1))
    std_dev = np.tile(std_dev.reshape(1, std_dev.shape[0], std_dev.shape[1]), (len(features), 1, 1))
    features = np.tile(features.reshape(features.shape[0], features.shape[1], 1), (1, 1, number_of_classes))

    # Feature probabilities
    feature_probabilities = gaussian(features, means, std_dev) # get the probability

    # Multiply all features' probabilities to get likelihood
    # Likelihood of each class
    # for c in range(number_of_classes):
    #     likelihood[c] = np.prod(feature_probabilities[np.nonzero(feature_probabilities[:, c]), c]) # mutliply for each feature 'Xi'
    # likelihood = np.prod(feature_probabilities, axis=1).reshape(-1, number_of_classes)
    likelihood = np.ma.prod(np.ma.masked_where(feature_probabilities==0, feature_probabilities), axis=1).filled(0).reshape(-1, number_of_classes)

    return likelihood


def naive_bayes_classify_optimized(NB_classifier, test_features):

    # NB_classifier is a dict with keys "prior", "means" and "std_dev"
    prior = NB_classifier["prior"]
    means = NB_classifier["means"]
    std_dev = NB_classifier["std_dev"]

    if len(test_features.shape) == 1:
        test_features = test_features.reshape(1, -1)

    likelihood = get_likelihood_optimized(test_features, means, std_dev)

    prior = np.tile(np.reshape(prior, (1, len(prior))), (len(test_features), 1))

    approx_posterior = likelihood * prior

    predicted_labels = np.argmax(approx_posterior, axis=1)

    return predicted_labels


def get_likelihood(features, means, std_dev):

    number_of_features, number_of_classes = means.shape

    # Feature probabilities
    # feature_probabilities = gaussian(tiled_features, means, std_dev) # get the probability
    feature_probabilities = np.zeros((number_of_features, number_of_classes))
    for f in range(number_of_features):
        for c in range(number_of_classes):
            feature_probabilities[f, c] = gaussian(features[f], means[f, c], std_dev[f, c])

    # Multiply all features' probabilities to get likelihood
    # Likelihood of each class
    likelihood = np.zeros((number_of_classes))
    for c in range(number_of_classes):
        likelihood[c] = np.prod(feature_probabilities[np.nonzero(feature_probabilities[:, c]), c]) # mutliply for each feature 'Xi'
    # likelihood = np.prod(feature_probabilities, axis=0)

    return likelihood


def naive_bayes_classify_unoptimized(NB_classifier, test_features):

    # NB_classifier is a dict with keys "prior", "means" and "std_dev"
    prior = NB_classifier["prior"]
    means = NB_classifier["means"]
    std_dev = NB_classifier["std_dev"]

    if len(test_features.shape) == 1:
        test_features = np.reshape(test_features, (1, len(test_features)))

    predicted_labels = []

    for t, test_sample_features in enumerate(test_features):
        # print("Progress: {0:0.04f}".format((t+1)/len(test_features)), end="\r")

        # Get the likelihood of the test samples belonging to each class
        likelihood = get_likelihood(test_sample_features, means, std_dev)

        # Calculate the approximate posterior = likelihood * prior
        approx_posterior = [np.asscalar(x*y) for x,y in zip(likelihood, prior)]
        #approx because of missing P(X) (constant) in the denominator

        # Make the prediction as that class with the maximum approximate posterior
        predicted_labels.append(np.argmax(approx_posterior))

    # print("")

    return np.array(predicted_labels)


def naive_bayes_classify(NB_classifier, test_features, optimized=True):

    if optimized:
        return naive_bayes_classify_optimized(NB_classifier, test_features)

    else:
        return naive_bayes_classify_unoptimized(NB_classifier, test_features)


def extract_2classes_with_binary_labels(i, j, X, Y):
    # Select class #0
    X_0 = X[Y == i]
    Y_0 = np.zeros((len(X_0)))
    # Select class #1
    X_1 = X[Y == j]
    Y_1 = np.ones((len(X_1)))
    # Join the two classes to make the set
    X_2classes = np.vstack((X_0, X_1))
    Y_2classes = np.append(Y_0, Y_1)
    return X_2classes, Y_2classes


# one-vs-one classifier
def one_vs_one_classifier(train_features, train_labels):
    clf = linear_model.SGDClassifier(random_state=1)
    clf.fit(train_features, train_labels)
    return clf


def multiclass_linear_classifier(train_features, train_labels):

    classes = np.unique(train_labels)
    number_of_classes = len(classes)
    number_of_features = train_features.shape[1]

    classifiers = []

    classes = np.unique(train_labels)
    number_of_classes = len(classes)
    train_labels = train_labels.reshape((train_labels.shape[0],))
    # For each pair of classes:
    for i in range(number_of_classes-1):
        for j in range(i+1, number_of_classes):

            # print("Training pair of classes:", i, j)

            # Extract the train features and labels of the two classes
            train_features_2classes, train_labels_2classes = extract_2classes_with_binary_labels(i, j, train_features, train_labels)

            # Let us make each one-vs-one classifier
            # Train the classifier on these features and labels
            clf = one_vs_one_classifier(train_features_2classes, train_labels_2classes)
            classifiers.append(clf)

    actual_labels = np.array([(i, j) for i in range(number_of_classes-1) for j in range(i+1, number_of_classes)]).T

    multiclass_linear_classifier = {}
    multiclass_linear_classifier["classifiers"] = classifiers
    multiclass_linear_classifier["actual_labels"] = actual_labels

    return multiclass_linear_classifier


def multiclass_linear_classify(multiclass_linear_classifier, test_features):

    classifiers = multiclass_linear_classifier["classifiers"]
    actual_labels = multiclass_linear_classifier["actual_labels"]

    if len(test_features.shape) == 1:
        test_features = np.reshape(test_features, (1, len(test_features)))

    # Find each classifier's prediction
    predicted_labels_from_all_classifiers = np.zeros((len(test_features), len(classifiers)), dtype=int)
    for c, clf in enumerate(classifiers):
        preds = np.asarray(clf.predict(test_features), dtype=int)
        predicted_labels_from_all_classifiers[:, c] = actual_labels[preds, c]

    # Take majority vote for each sample
    predicted_labels = []
    for p in predicted_labels_from_all_classifiers:
        predicted_labels.append(np.argmax(np.bincount(p)))

    return np.array(predicted_labels)
