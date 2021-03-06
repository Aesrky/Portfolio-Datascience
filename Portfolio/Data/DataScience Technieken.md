In dit stuk worden de technieken die ik heb toegepast voor de minor Data Science aan de Haagse Hogeschool uitgelegd. Tevens wordt er per stuk de theorie achter de keuze uitgelegd en waarom er gekozen is voor deze technieken.
Daarbij komen de volgende aspecten terug: 

* Predictive Modelling
* Data Preparation
* Data Visualization
* Data Collection
* Data Evaluation


Veelal is dit in hulp gemaakt met behulp van klasgenoten aangezien ik niet de meest beste programmeur ben. Sommige stukken code zijn door mijzelf gemaakt in een poging tot het beheersen van de stof en om enigszins de stof te kunnen toepassen.

Voor onze opdract hebben wij onder andere gebruikt gemaakt van het concept Predictive Modeling.
Het was in eerste instantie namelijk niet duidelijk wat men precies voor technieken kon gebruiken voor de dataset gegeven door het CBS. 
Grof gezegd hebben wij middels deze techniek getracht patronen of gelijkenissen in de dataset proberen te vinden.

Ten eerste hebben ik en mijn collega's de data handmatig gelabeld, hierbij hebben wij de data een 1,2,3,4 of classificatie meegegeven. 
Vervolgens is er een preprocess over de data toegepast.

Dit kan uitgevoerd worden in 4 stappen:

* Smoothen van Noisy data (Dit gedeelte was niet van toepassing op onze dataset)
* Aggregeren van Data - Het in een leesbare tabel zetten van verkregen e-mail data 

Hiervoor heb ik gebruik gemaakt van de package pandas. Dit is geleerd op de courses van datacamp:

```python
import pandas as pd


# Data laden van een .csv file
data = pd.DataFrame.from_csv('cbs.data')
# Converteren van data van een string naar tijd
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)
```

Dit was een stukje die ik had toegepast om in ieder geval de data van CBS e-mails in een datum formaat te zetten zodat het duidelijk was welke email wanneer is gestuurd.


* Het invoeren van data waarbij niets ingevoerd is. Meestal wordt er door een script gekeken waar data leegstaat.
Alle data die het model als <1 herkent wordt vervangen met een 0. Soms wordt het vervangen door een NaN = Not a Number.

Bij het zogeheten cleanen en voorbereiden van data heb ik een aantal tutorials gevolgd waarbij naar voren kwam hoe men leegstaande cellen zo goed mogelijk kon aanpakken. 
Daarbij is door mij de volgende code gehanteerd:

```python
# Lijst van alle leegstaande waarde
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("cbs.csv", na_values = missing_values)
```

Zoals in de comment staat, zorgt dit stukje code ervoor dat het een lijst maakt van de dataset waarbij alle data die leegstaat wordt geinventariseerd.
Vervolgens heb ik gekozen om leegstaande vakken te vervangen door een nummer met de volgende code:

```python
# Leegstaande waarde veranderen door een nummer
df['cbs].fillna(125, inplace=True)
```
Dit zorgt ervoor dat men bij het debuggen uren werk minder hoeft te doen wanneer je een diagnose / analyse gaat uitvoeren op je data.

* Het verwijderen van Data punten die niet in contrast staat met de overige data.


Vervolgens hebben wij 3 soorten algoritme gebruikt die het beste bij ons data paste. Dit waren de volgende algoritme:

* Multinomial Bayes
* Complement Bayes
* Logistic Regression

Deze 3 algoritme hebben wij "getrained" over ons dataset. Daarbij behoort onder andere de volgende code:

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


Tevens wilde wij ook weten hoe belangrijk bepaalde woorden zijn binnen een dataset. Zogeheten Word Embeddings, deze geven aan tekst een bepaalde waarde in nummers. Hoe belangrijker een bepaald stuk tekst, des te hoger de waarde in nummers.
Zo kwamen wij uit op:

* TF - IDF Ngram = Hoe belangrijk een woord is in een document of in een collectie van documenten
```python
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
```



* Count Vectors
```python
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

```


Foto resultaat model neigt naar type 3 staat in rapport resultaten gelijk

![Test](/Portfolio/Courses/Screenshot%202019-01-11%20at%2010.40.46.png)


* Ngram = Model over de relatie tussen woorden. Daarbij creeert het bijvoorbeeld, Unigram(1 woord), Bigram(2 woorden), Trigram (3 woorden) etc.
In het geval van een bigram kunnen we meegeven dat 2 bepaalde woorden bij elkaar een bepaalde opbouw van een zin aangeven bijvoorbeeld.

Per model wordt er een onderscheid gemaakt tussen Training, Test en Cross validation set van de data. Te zien in de 2 stukjes code hierboven. Dit is geleerd op de Coursera Course van Andrew NG.
* Training set is het initiele set waarbij een model of algoritme op wordt "gefit". Als dit op een goede manier is toegepast kan de validation set observaties of responses voorspellen.
* Een validatieset is gesplitte data set en ook een unbiased evaluatie model. Men kan ook snel zien op de validatieset wanneer er sprake is van een overfit. Door middel van bepaalde handelingen zoals regularization kan dit probleem worden aangepakt.
* Tot slot wordt er een test dataset geproduceerd. Dit is ook een unbiased evaluatie maar van het "eind" model. Tevens is dit een onafhankelijk model van de training set, maar volgt het wel dezelfde distributie. 
Als een model op de training set en de test set een goede fit heeft, dan kan dit betekenen dat er sprake is van een minimale overfit.
Als een model beter op de training set past dan op de test set, is er meestal sprake van een overfit.

De verdeling van Training, Cross en Test wordt veelal voorgeschreven als: 60% de training set, 20% cross validation set en 20% de test set.

* Overfit
* Underfit


Error Analyse Code & Confusion Matrix Visualisatie 

```python
def create_confusion_matrix(self, valid_y, predictions_valid, model_name):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(valid_y, predictions_valid)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['Beschikbaarheidsvraag', 'Verduidelijkingsvraag', 'Niet relevant', 'Relevante query vraag']
    self.plot_confusion_matrix(cnf_matrix,
                               classes=class_names,
                               title=model_name + ' Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                               normalize=True,
                               title=model_name + ' Normalized confusion matrix')

    plt.show()
```
![test1](/Portfolio/Courses/Screenshot%202019-01-11%20at%2011.07.11.png)
![test2](/Portfolio/Courses/Screenshot%202019-01-11%20at%2011.07.31.png)

Resultaten uit de Confusion Matrix met bovenstaande code. Uit deze confusion matrix heb ik vervolgens een error analyse gemaakt te lezen in het rapport.