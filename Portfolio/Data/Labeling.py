from sklearn import multiclass
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, linear_model, model_selection, svm, feature_selection, neighbors, ensemble, \
    naive_bayes
from sklearn.metrics import precision_recall_fscore_support as score

import xgboost
import pandas as pd
from Exporter.Exporter import Exporter
from PreProcessor import PreProcessor
from Visualization.Plot_functions import PlotFunctions
from Models.BaseModel import BaseModel
from Importer import Importer

from hpsklearn import HyperoptEstimator, any_classifier, any_sparse_classifier, tfidf, multinomial_nb
from hyperopt import hp


class MultiClassifier(BaseModel):
    def __init__(self, trainDF):
        super().__init__()
        prePro = PreProcessor()
        self.pf = PlotFunctions()
        self.trainDF = trainDF
        self.X_train, self.X_test, self.y_train, self.y_test = \
            prePro.split_train_test(trainDF['cleaned_sentence'], trainDF['classification'], 0.4)
        self.X_test, self.X_cross, self.y_test, self.y_cross = \
            prePro.split_train_test(self.X_test, self.y_test, 0.5)

        self.all_scores = list()
        self.models = {
            'MultinomialNB': naive_bayes.MultinomialNB(alpha=0.767, class_prior=None, fit_prior=True),
            'ComplementNB': naive_bayes.ComplementNB(alpha=0.767, class_prior=None, fit_prior=True),
            'LogisticRegression': linear_model.LogisticRegression(solver='lbfgs')
        }

    def check_model(self, classifier, X, y, model_name, amount_features, feature_name, datatype):
        predictions = self.funcs.return_prediction_data(classifier,
                                                        X)
        classifications = y.reset_index(drop=True)
        precision, recall, f_score, true_sum = score(classifications,
                                                     predictions,
                                                     average='weighted')
        self.all_scores.append({'model_name': model_name,
                                'feature_name': feature_name,
                                'amount_features': amount_features,
                                'precision': precision,
                                'recall': recall,
                                'f1_score': f_score,
                                'datatype': datatype
                                })

        # Create a confusion matrix:
        # if datatype == 'cross':
        #     pf = PlotFunctions()
        #     pf.create_confusion_matrix(classifications, predictions,
        #                                '{0} {1} {2}'.format(model_name, feature_name, amount_features))

    def count_vectors(self, features):
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_df=1.0, max_features=features)
        count_vect.fit(self.trainDF['cleaned_sentence'])
        xtrain_count = count_vect.transform(self.X_train)
        xvalid_count = count_vect.transform(self.X_test)
        xcross_count = count_vect.transform(self.X_cross)

        for model_name, model in self.models.items():
            mc_model = multiclass.OneVsRestClassifier(model)
            classifier = mc_model.fit(xtrain_count, self.y_train)

            # Training predictions
            self.check_model(classifier, xtrain_count, self.y_train, model_name, features, 'count_vectors', 'training')

            # Test predictions
            self.check_model(classifier, xvalid_count, self.y_test, model_name, features, 'count_vectors', 'test')

            # Cross Validation predictions
            self.check_model(classifier, xcross_count, self.y_cross, model_name, features, 'count_vectors', 'cross')

    def tfidf_words(self, features):
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=features)
        tfidf_vect.fit(self.trainDF['cleaned_sentence'])
        xtrain_tfidf = tfidf_vect.transform(self.X_train)
        xvalid_tfidf = tfidf_vect.transform(self.X_test)
        xcross_tfidf = tfidf_vect.transform(self.X_cross)

        for model_name, model in self.models.items():
            mc_model = multiclass.OneVsRestClassifier(model)
            classifier = mc_model.fit(xtrain_tfidf, self.y_train)

            # Training predictions
            self.check_model(classifier, xtrain_tfidf, self.y_train, model_name, features, 'tfidf_words', 'training')

            # Test predictions
            self.check_model(classifier, xvalid_tfidf, self.y_test, model_name, features, 'tfidf_words', 'test')

            # Cross Validation predictions
            self.check_model(classifier, xcross_tfidf, self.y_cross, model_name, features, 'tfidf_words', 'cross')

    def tfidf_ngram(self, features):
        tfidf_vect_ngram = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 5), max_features=features)
        tfidf_vect_ngram.fit(self.trainDF['cleaned_sentence'])
        xtrain_tfidf = tfidf_vect_ngram.transform(self.X_train)
        xvalid_tfidf = tfidf_vect_ngram.transform(self.X_test)
        xcross_tfidf = tfidf_vect_ngram.transform(self.X_cross)

        for model_name, model in self.models.items():
            mc_model = multiclass.OneVsRestClassifier(model)
            classifier = mc_model.fit(xtrain_tfidf, self.y_train)

            # Training predictions
            self.check_model(classifier, xtrain_tfidf, self.y_train, model_name, features, 'tfidf_ngram', 'training')

            # Test predictions
            self.check_model(classifier, xvalid_tfidf, self.y_test, model_name, features, 'tfidf_ngram', 'test')

            # Cross Validation predictions
            self.check_model(classifier, xcross_tfidf, self.y_cross, model_name, features, 'tfidf_ngram', 'cross')

    def tfidf_char(self, features):
        tfidf_vect_ngram_chars = TfidfVectorizer(
            analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=features)
        tfidf_vect_ngram_chars.fit(self.trainDF['cleaned_sentence'])
        xtrain_tfidf = tfidf_vect_ngram_chars.transform(self.X_train)
        xvalid_tfidf = tfidf_vect_ngram_chars.transform(self.X_test)
        xcross_tfidf = tfidf_vect_ngram_chars.transform(self.X_test)

        for model_name, model in self.models.items():
            mc_model = multiclass.OneVsRestClassifier(model)
            classifier = mc_model.fit(xtrain_tfidf, self.y_train)
            # Training predictions
            self.check_model(classifier, xtrain_tfidf, self.y_train, model_name, features, 'tfidf_char', 'training')

            # Test predictions
            self.check_model(classifier, xvalid_tfidf, self.y_test, model_name, features, 'tfidf_char', 'test')

            # Cross Validation predictions
            self.check_model(classifier, xcross_tfidf, self.y_cross, model_name, features, 'tfidf_char', 'cross')

    def get_and_print_all_scores(self):
        print('Running for count_vectors')
        for i in range(500, 5000, 500):
            self.count_vectors(i)
            self.tfidf_words(i)
            self.tfidf_ngram(i)
            self.tfidf_char(i)


imp = Importer()
trainDF = imp.ImportFuncs.read_csv_into_dataframe(
    'csv_classification/Multi-class/classified_sentences_all.csv')

prePro = PreProcessor()
trainDF = prePro.clean_dataframe_for_training(trainDF)
print(trainDF.head())

a = MultiClassifier(trainDF)
a.get_and_print_all_scores()

print(a.all_scores)

exp = Exporter()
exp.create_csv_scores(a.all_scores, 'all_scores_cleaned')
