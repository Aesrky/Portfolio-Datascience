Voor onze dataset hebben wij gebruikt gemaakt van het concept Predictive Modeling oftewel Data Mining.
Het is in eerste instantie namelijk niet duidelijk wat men precies voor technieken kan gebruiken voor de dataset gegeven door het CBS. 
Grof gezegd hebben wij middels deze techniek getracht patronen of gelijkenissen in de dataset proberen te vinden.

Ten eerste hebben wij de data handmatig gelabeld, hierbij hebben wij de data een 1,2,3,4 of classificatie meegegeven. 
Vervolgens hebben wij de data Ge-Preprocessed. Meer over dit proces in het kopje "Data Preparation".

Vervolgens hebben wij 3 soorten algoritme gebruikt die het beste bij ons data paste. Dit waren de volgende algoritme:

* Multinomial Bayes
* Complement Bayes
* Logistic Regression

Deze 3 algoritme hebben wij "getrained" over ons dataset. Daarbij behoort de volgende code:

```python

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

```



Tevens wilde wij ook weten hoe belangrijk bepaalde woorden zijn binnen een dataset. Zogeheten Word Embeddings, deze geven aan tekst een bepaalde waarde in nummers. Hoe belangrijker een bepaald stuk tekst, des te hoger het nummer.
Zo kwamen wij uit op:

* TF - IDF = Hoe belangrijk een woord is in een document of in een collectie van documenten
* Ngram = Model over de relatie tussen woorden. Daarbij creeert het bijvoorbeeld, Unigram(1 woord), Bigram(2 woorden), Trigram (3 woorden) etc.
In het geval van een bigram kunnen we meegeven dat 2 bepaalde woorden bij elkaar een bepaalde opbouw van een zin aangeven bijvoorbeeld.
* Count Vector