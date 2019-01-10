In dit stuk worden de technieken voor de minor Data Science aan de Haagse Hogeschool toegepast met uitleg waarom er gekozen is voor deze technieken en hoe deze zijn toegepast.
Daarbij komen de volgende aspecten terug: 

* Predictive Modelling
* Data Preparation
* Data Visualization
* Data Collection
* Data Evaluation


Veelal is dit in hulp gemaakt met behulp van klasgenoten aangezien ik niet de meest beste programmeur ben. Sommige stukken code zijn door mijzelf gemaakt in een poging tot het beheersen van de stof en om enigszins de stof te kunnen toepassen.

Voor onze dataset hebben wij gebruikt gemaakt van het concept Predictive Modeling oftewel Data Mining.
Het is in eerste instantie namelijk niet duidelijk wat men precies voor technieken kan gebruiken voor de dataset gegeven door het CBS. 
Grof gezegd hebben wij middels deze techniek getracht patronen of gelijkenissen in de dataset proberen te vinden.

Ten eerste hebben wij de data handmatig gelabeld, hierbij hebben wij de data een 1,2,3,4 of classificatie meegegeven. 
Vervolgens hebben wij de data Ge-Preprocessed.

Dit kan uitgevoerd worden in 4 stappen:

* Smoothen van Noisy data (Dit gedeelte was niet van toepassing op onze dataset)
* Aggregeren van Data - Het in een leesbare tabel zetten van verkregen e-mail data 

Hiervoor is gebruik gemaakt van de package pandas. Dit is geleerd op de courses van datacamp:

```python
import pandas as pd
import dateutil

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
Dit zorgt ervoor bij het debuggen dat het je uren kan schelen bij het analyseren van je debug.

* Het verwijderen van Data punten die niet in contrast staat met de overige data.



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

Per model is er de Training - Test - Cross 




* Ngram = Model over de relatie tussen woorden. Daarbij creeert het bijvoorbeeld, Unigram(1 woord), Bigram(2 woorden), Trigram (3 woorden) etc.
In het geval van een bigram kunnen we meegeven dat 2 bepaalde woorden bij elkaar een bepaalde opbouw van een zin aangeven bijvoorbeeld.
* Count Vector