# Portfolio-Datascience

<h1>Persoonlijke Portfolio</h1>

Het persoonlijke portfolio voor de minor Data Science aan de Haagse Hogeschool

* <b>Naam</b>: Askin Sarikaya
* <b>Studentnummer</b>: 14121409
* <b>E-mail</b>: 14121409@student.hhs.nl


# Table of contents
1. [Introductie](#Introductie)
2. [Domain Knowledge](#Domein)
    1. [Jargon](#Jargon)   
3. [Courses](#Courses)
   1. [Datacamp Courses](#Datacamp)
   2. [Coursera Courses](#Coursera)
4. [Data Science Technieken](#Data)
    1. [Predictive Modeling](#Predictive)
    2. [Data Preparation](#Preparation)
    3. [Data Visualization](#Visualization)
    4. [Data Collection](#Collection)
    5. [Data Evaluation](#Evaluation)
    6. [Diagnostics](#Diagnostics)
5. [Communicatie](#Communicatie)
    1. [Presentaties](#Presentaties)
    2. [Paper](#Paper)
    3. [Scrum](#Scrum)
    4. [Reflectie](#Reflectie)



<h1> Leeswijzer </h1>

Voor de herkansing heb ik twee wijzigingen moeten toebrengen aan mijn portfolio:
- Datacamp State of Accomplishment toevoegen aan courses
- Showcase van mijn "machine learning skills" die ik heb geleerd in vorm van 2 notebooks.

Voor de herkansing heb ik 2 notebooks gemaakt met data van Kaggle om mijn machine learning skills te "showcasen". Zodat ik in ieder geval kan laten zien dat ik de basis begrijp en een model kan trainen op een willekeurige dataset. Bij de 2 notebooks heb ik gekozen om verschillende methodieken en technieken toe te passen. 

Mijn taken waren vooral om te kijken hoe ik een lineaire regressie model kon fitten & initialiseren op random data, maar ook hoe ik data zover kon krijgen dat het betekenisvol werd. Daarvanuit heb ik een 2e notebook gemaakt waarin ik meerdere technieken heb toegepast dan alleen maar een vorm van lineaire regressie. In het hoofdstuk Data heb ik de 2 notebooks met een aantal stappen toegelicht. In de 2 notebooks zelf heb ik geprobeerd hoofdstukken (of code line comments) te plaatsen met tussentekst om te laten zien wat ik per stap heb toegepast. 

<b>Technieken & Modellen kort:</b>

- Lineaire Regressie
- Checken wanneer sprake is van een overfit of underfit (Diagnostics)
- PCA (Principle Component Analyse)
- Decision Tree Classifier (Feature Selectie)
- Splitten Test - Train Data, Trainen van model & fitten van model d.m.v. lineaire regressie
- Feature Engineering (o.a. maken van nieuwe features van bestaande features)
- Explatory Data Analyse
- Checken van missende data in de dataset
- Scheiden van Continue & Categorische Variabele & Inhoud checken van deze waarde
- Controleren Type Datasets 
- Data Visualisatie (Box plots, scatter plots, histogram etc.)
- Detecteren outliers
- Correlatiematrix (correlatie tussen elke feature)
- Feature Importance
- Feature Selectie met Lasso Regressie
- Predictive Modeling met meest belangrijke features & alle features
- Feature Scaling + PCA (hierboven genoemd)
- Trainen, Test & Fitten op verschillende modellen om o.a. accuracy score te bepalen van CoVariance, Logistic Regression, Decision Tree, Random Forest, SVM. Om vervolgens te bepalen welke classifier het beste werkt op mijn data (in dit geval Lineair Regression)
- Checken van de Mean Squared Error & Variance + Accuracy
- Predicten van classificatie rapport. (Precision, Recall, F1-score) & Dit vervolgens Plotten.
- Area under the Curve predicten

&nbsp;


- [Klik hier voor Hoofdstuk Data](#Data)


- [Notebook 1](/Portfolio/Notebooks/Askin%20Sarikaya%20Lineaire%20Regressie%20Showcase.ipynb) 

- Kaggle Dataset link:
https://www.kaggle.com/shivachandel/kc-house-data

- [Notebook 2](/Portfolio/Notebooks/Askin%20Sarikaya%20ShowCase%20All%20Skills.ipynb) 

- Kaggle Dataset link: https://www.kaggle.com/janiobachmann/bank-marketing-dataset

&nbsp;

<details><summary><h2>Klik hier voor Introductie</h2></summary>
<p>
Voor onze minor Data Science aan de Haagse Hogeschool hebben wij in een groep van 4 mensen een opdracht uitgevoerd namens het CBS.
Daarbij werd er getracht de top 10 categorieen, van gegeven data van het CBS, eruit te filteren.
Data die door het CBS beschikbaar is gesteld en tevens gebruikt is voor deze opdracht zijn de gestelde vragen per e-mail aan het CBS.
Omdat dit zodanig groot bleek te zijn, door het aantal categorieen waarin vragen kunnen worden gesteld meer dan 180 bleek, hebben wij ons als groep beperkt tot 2 datasets:

* Inkomen
* Bevolkingsgroep

Deze datasets hebben wij handmatig gelabeld en vervolgens geprobeerd om ons algoritme op los te laten. Daarbij hebben wij voornamelijk volgende modellen gebruikt:

* Multionomial Naive Bayes
* Complement Naive Bayes
* Logistic Regression

Dit waren ook tevens de modellen met het beste resultaat op onze skewed dataset. 
</p></details>


<h1>Domein</h1>
Voordat ik aan de course begon had ik nauwelijks kennis op het gebied van Data Science. Het werd vaak in mijn omgeving genoemd als de "next" big thing en ik ben als het ware gestapt op de hype train.
Daarbij ging ik met niet al teveel programmeer kennis in dit vak en daarbij had ik geen affiniteit met wiskunde tijdens mijn middelbare school tijden. 
De keuze viel al snel op de chatbot van het CBS. Dat werd toch uiteindelijk heel iets anders dan verwacht. Wel heb ik gigantisch veel kennis opgedaan met python en machine learning concepten door veel te onderzoeken en te relateren aan ons project.

<details><summary><h3>Klik hier voor inhoud hoofdstuk Domein</h3></summary><p>
<h2>Jargon</h2>

* Naive Bayes = Naive Bayes is een familie van simpele "probabilistic classifier" gebaseerd op de Bayes' theorem met een sterke (naief) onafhankelijkheid veronderstellingen tussen de features
* Machine Learning = Is het wetenschappelijke onderzoek van algoritme en statistische modellen die computers gebruiken om progressieve verbetering te boeken op de prestaties van een specifieke taak.
* Skewed Dataset = Ongebalanceerde dataset tussen verschillende classen
* Logistic Regression = Logistische Regressie binnen de statistiek wordt gebruikt om een dichotome uitkomstvariabele te relateren aan een of meerdere variabelen
* Feature = De input gegeven aan een predictive model. 
* Data Cleaning = Het verschonen en categoriseren van de dataset
* TF - IDF Ngram = Hoe belangrijk een woord is in een document of in een collectie van documenten
* Ngram = Model over de relatie tussen woorden. Daarbij creeert het bijvoorbeeld, Unigram(1 woord), Bigram(2 woorden), Trigram (3 woorden) etc. In het geval van een bigram kunnen we meegeven dat 2 bepaalde woorden bij elkaar een bepaalde opbouw van een zin aangeven bijvoorbeeld.
* Data Visualisatie = Het visualiseren van data om zo een beeld te geven van je resultaten of bevindingen
* Data Manipuleren = Het veranderen van metadata van je dataset
* Data Importeren = Het importeren van Data
* Data Preparatie = Het cleanen van "raw data" om als input te dienen voor een predictive model
* Pentesting = Het vinden van kwetsbaarheden in systemen
* OneVsRest = Het kiezen van een class en het trainen van een two class classifier met de samples van een geselecteerde class aan de ene kant en alle andere voorbeelden aan de andere kant.
</details></p>



 <h1>Courses</h1>
 
In dit hoofdstuk worden de benodigde en extra opdrachten die gemaakt zijn toegelicht.

<details><summary><h3>Klik hier voor inhoud hoofdstuk Courses</h3></summary>
       <p>
<h2>Datacamp</h2>

Alle benodigde opdrachten beschreven in de wekelijkse agenda voor datacamp zijn voltooid. 
Hieronder wordt per course een korte stuk beschreven over de toegevoegde waarde van de course ten behoeve van mijn ontwikkeling.

Voor bewijs van voltooiing refereer ik graag naar mijn Datacamp Account:

[Datacamp Account](https://www.datacamp.com/profile/14121409)



* <b> Programmeren </b> (19100 XP)
1. (Course) [Introduction to Python](https://www.datacamp.com/courses/intro-to-python-for-data-science) ,  [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/5afe33bcab85a7d1bdb1b8309ad8819bfe8e252b) 
2. (Course) [Intermediate Python for Data Science](https://www.datacamp.com/courses/intro-to-python-for-data-science) , [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/785552bb917d2f0ca7c12f0aa2f425aa2d72cfd6) 
3. (Chapter) [Writing your own functions](https://www.datacamp.com/courses/python-data-science-toolbox-part-1) ,  (Chapter dus geen SOA)
4. (Chapter) [Default arguments, variable-length arguments and scope](https://www.datacamp.com/courses/python-data-science-toolbox-part-1)  , (Chapter dus geen SOA)
5. (Course) [Python Data Science Toolbox (Part 2)](https://www.datacamp.com/courses/python-data-science-toolbox-part-2) , [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/569c578fd70f48c9602df640317c8260e2ac5c57)  

Het leren van het programmeren van Python heeft er tot bijgedragen dat ik op een basis-gevorderd niveau Python code kan begrijpen. 
Dit zorgde direct ervoor dat bij het lezen van de code gemaakt (vooral tutorials gevolgd op internet) door de programmeurs van de groep, de code niet vreemd overkwam bij mij op persoon.
Dit zorgde er tevens voor dat ik gericht ideeen/feedback kon geven passend op de haalbaarheid van de opdracht zonder dat dit buiten proporties kwam te liggen.

Boven alles heeft dit ervoor gezorgd dat ik kundiger met Linux en tegelijk met Python ben geworden. Dit heeft ervoor gezorgd dat ik in mijn eigen vakgebied (IT - Security) scripts kan begrijpen & schrijven als ik bijvoorbeeld iets aan het pentesten ben.
Deze relatie had ik voorheen niet kunnen leggen en dit is zodoende ook een grote positieve bijdrage in mijn carriere.


* <b> Importeren en Cleanen van Data </b> (8020 XP)
1. (Course) [Importing Data in Python (Part 1)](https://www.datacamp.com/courses/importing-data-in-python-part-1) & Mandatory , [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/6dbaa8c404b2a07c22554880d962f1dbe56f6444)   
2. (Chapter) [Introduction and flat files](https://www.datacamp.com/courses/importing-data-in-python-part-1) , [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/6dbaa8c404b2a07c22554880d962f1dbe56f6444) 
3. (Course) [Cleaning data in Python](https://www.datacamp.com/courses/cleaning-data-in-python) ,  [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/2227bd9973cb5f48ed20f5bd6c8f5cf0fff47a67)   

Dit gedeelte van Python heeft ervoor gezorgd dat ik nu weet hoe ik verschillende soorten data kan invoeren in python, en vervolgens met deze data aan de slag kan gaan.
Het lijkt iets simpels, maar het invoeren van data was voorheen al een hele uitdaging. Bovendien heeft het een beeld gegeven hoe ik met mijn blote oog op zoek moet gaan naar nieuwe "raw" data. Tevens heeft deze course ook bijgedragen aan het leren van cleanen van coding met de pandas package.
Dit heb ik in kleine schaal toegepast in de hoofdstuk Data Preparation. Tot slot heeft dit mij een algehele beeld gegeven van hoe ik data kan importeren en cleanen om het bruikbaar te maken voor mijn predictive models.


* <b> Data Manipulatie </b> (2080 XP)
1. (Chapter) [Data ingestion & inspection](https://www.datacamp.com/courses/pandas-foundations) (Chapter dus geen SOA)
2. (Chapter) [Exploratory Data Analysis](https://www.datacamp.com/courses/pandas-foundations) (Chapter dus geen SOA)

In deze chapters heb ik kort geleerd hoe ik met Pandas om kan gaan en heb ik ingezien wat een krachtige tool dit kan zijn. Ik heb dit echter niet toegepast binnen mijn project.

* <b> Data Visualisatie </b> (3520 XP)
1. (Chapter) [Plotting 2D Arrays](https://www.datacamp.com/courses/introduction-to-data-visualization-with-python) (Chapter dus geen SOA)
2. (Chapter) [Statistical plots with Seaborn](https://www.datacamp.com/courses/introduction-to-data-visualization-with-python) (Chapter dus geen SOA)
3. (Chapter) [Customizing Plots](https://www.datacamp.com/courses/introduction-to-data-visualization-with-python) (Chapter dus geen SOA)

Mijn favoriete onderdeel en volgens mij ook het onderdeel wat ik dagelijks gebruik voor mijn eigen werk. Ik heb hierbij geleerd hoe ik mijn grafieken zo goed mogelijk kan visualiseren en heb zelfs Excel afgezworen na het volgen van deze tutorials.
Ik gebruik de package seaborn nog dagelijks voor het maken van heatmaps voor mijn werk. Visualisatie is echt key als dit zo goed mogelijk de data vertegenwoordigd. Dit heb ik geleerd van deze course en zal het zonder twijfel een leven lang meedragen.


* <b> Waarschijnlijkheid & Statistiek </b> (4350 XP)
1. (Course) [Statistical Thinking in Python (Part 1)](https://www.datacamp.com/courses/statistical-thinking-in-python-part-1)
[Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/21b5a596590b72f3294c0eacd2395525068533dd)  

Een course waarbij wordt weergegeven hoe je statistiek kan vertalen binnen Python. Een cruciale stap binnen Data Science, want je kan zoveel moeite doen om je data te vergaren en het vormen in een product waar je mee kan werken. 
Het zal je niets opbrengen als je geen duidelijke conclusies van je data kan trekken. Deze course heeft geholpen bij het formuleren van duidelijke resultaten en conclusies voor de paper.


* <b> Machine Learning </b> (4300 XP)
1. (Course) [Supervised Learning with scikit-learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn) , [Statement of Accomplishment](https://www.datacamp.com/statement-of-accomplishment/course/31a1deb517b91e130707e9fe011c280e588a5d3f) 

De Course die mij heeft geholpen met het maken van een predictive model. Hierbij heb ik informatie vergaard over classificatie, regressie maar ook het fine tunen van mijn model wat cruciaal is.
Mijn code voor predictive modelling is veelal gebaseerd op stukjes van de course.


<h2>Coursera</h2>


Voor Coursera zijn de weken 1,2,3 en 6 afgerond. De bijbehorende opdrachten zijn daarbij niet gemaakt. 
Alle quiz onderdelen zijn met een voldoende afgerond.

![Coursera](/Portfolio/Courses/Timeline%20Coursera.png)


De video's van Coursera en de bijbehorende quiz hebben bijgedragen tot een betere kennis op het gebied van machine learning. 
Bij Datacamp lag de focus meer op het toepassen en programmeren. Bij Coursera werd de achterliggende gedachte, formules etc. het gehele concept uitgelegd over machine learning dus ook toepassen.
Dat ik de stof begreep getuigd ook van mijn voldoende op de toets. Tevens zorgde het bestuderen van Coursera ervoor dat ik nieuwe ideeen opdeed en dit zorgde er direct voor dat bepaalde handelingen veel makkelijker konden uitgevoerd. Immers, wij begrepen het concept van machine learning veel beter nu.
&nbsp;

</details></p>

<h1>Data</h1>

Voor de herkansing heb ik gekozen om 2 notebooks te maken om mijn zogeheten "skills" te laten zien. Deze notebooks bevatten in ieder geval de benodigde requirements die men acht te begrijpen na de datascience minor.


<h2> Python Notebook 1 </h2>

* [Notebook 1](/Portfolio/Notebooks/Askin%20Sarikaya%20Lineaire%20Regressie%20Showcase.ipynb)

<b>Welke Dataset:</b>

Voor deze notebook / showcase van geleerde technieken heb ik gebruik gemaakt van een Kaggle dataset. Daarbij is bewust gekozen voor een dataset zonder Kernels.
Het betreft een dataset met historische data van huizen die verkocht zijn in de staat Washington (USA). In de periode van mei 2014 tot mei 2015. 

Kaggle Dataset link: https://www.kaggle.com/shivachandel/kc-house-data 

<b>Stappen, Taken, Technieken & Modellen die ik heb gedemonstreerd op de Huis Dataset:</b>


•	Data cleaning waarbij ik tabellen het “gedropt” die niet benodigd zijn voor het trainen van de dataset. Tevens heb ik ook bekeken of er geen missende waarde bevatten in de dataset.

•	Vervolgens heb ik op twee manieren gevisualiseerd wat de correlatie is tussen features. 
    o	Een Panda’s Dataframe waarbij de correlatie te zien is in nummers
    o	Een Correlatie heatmap
    
•	Omdat ik de dataset wil “trainen” op de feature price heb ik bekeken welke feature de sterkste correlatie heeft met deze feature. Dit was feature: 
    o	“sqft_living15”
    o	“sqft_living”
Deze twee features heb ik vervolgens geplot in de vorm van een scatter plot.

•	Ook heb ik gekeken of de andere features die enigszins hoog scoren een sterke correlatie hebben met elkaar. Hieruit vloeit meerdere tabellen.
•	Vervolgens heb ik van de “price” colum een variabele (y) gemaakt om dit te trainen. 

•	Na het maken van de variabele heb ik de data gesplitst in een training en test data. Waarbij ik een standaard test data size heb gehanteerd (33%).  En gekeken naar de accuracy score van lineaire regressie op de data. 

•	Vervolgens heb ik de lineaire regressie model geinitialiseerd en vervolgens deze “gevoed” in de training data X voor training data Y. Ons model is nu getrained op de gegeven data.

•	Tot slot heb ik gekeken welke feature de meest belangrijke is d.m.v. regressie coëfficiënt. Deze heb ik vervolgens in een panda dataframe geplot om te zien welke feature de meest belangrijke is. 

•	Dit bleek feature waterfront te zijn. Eerst heb ik van de feature waterfront een numpy array gemaakt. Vervolgens heb ik een object gecreëerd (LinearRegression() voor de klasse). Tot slot heb ik de lineaire regressie uitgevoerd. Het bleek jammer genoeg underfit te zijn. 

<h2>Python Notebook 2 </h2>

* [Notebook 2](/Portfolio/Notebooks/Askin%20Sarikaya%20ShowCase%20All%20Skills.ipynb)

<b>Welke Dataset:</b>
De data is gerelateerd aan een marketing campagne van een Portugese bank instituut. Deze campagnes zijn gebaseerd op telefoongesprekken. Ook was er sprake van meer dan een keer contact met de zelfde client. Dit was benodigd om het product(Deposit colum in data) een waarde van "Yes" of "No" te geven.

* Kaggle Dataset link: https://www.kaggle.com/janiobachmann/bank-marketing-dataset

<b>Stappen, Taken, Technieken & Modellen die ik heb gedemonstreerd  op de Bank Dataset:</b>


•	Mijn data heeft heel veel features (kolommen). Van die features hebben aantal een aantal categorische (ordinaal) waarde en sommige een continue waarde. Ik heb een code geïmplementeerd die deze twee waarde teruggeeft in lijsten: CON = bevat namen van kolommen met de continue waarde als inhoud & CAT = bevat namen van kolommen die een categorische waarde bevatten. Ik heb dit gedaan omdat ik na deze stap. Het automatiseringsproces van een aantal stappen kan vergemakkelijken. Ik hoef dus niet handmatig meer data te analyseren tot een bepaalde punt.

•	Data analyse heb ik toegepast op de dataset. Ik heb hierbij gekeken naar missende data in de dataset. Ook EDA(Explatory Data Analyse) heb ik toegepast. Waarbij ik een aantal features heb gevisualiseerd en waardevolle insights heb kunnen verkrijgen

•	Door het plotten van boxplots heb ik de outliers kunnen observeren, zo weet ik dat de data dit bevat.

•	Met feature engineering heb ik nieuwe features gecreëerd van al bestaande features. De data heeft een groot aantal features om het model te trainen. Nieuwe features kan gemaakt worden van bestaande features zoals, leeftijd(age). Daarbij heb ik dummy variabeles gemaakt van de ordinale waarde. Dit heb ik gedaan omdat de machine learning model geen string waardes begrijpt. Bijvoorbeeld in de gender kolom waar te zien is dat de feature een aantal stringwaarde bevat zoals “male” of “female”. Dit heb ik geconverteerd in een kolom genaamd “male” met een 1 en 0 waarde (binary). Als de waarde 1 is, is het een “male”, zo niet een “female” in dit geval.

•	De correlatiematrix (heatmap) laat de correlatie zien tussen elke feature. Features die een hoge correlatie bevatten zorgen ervoor dat er sneller sprake is van data redundancy.

•	Met feature selection heb ik  de meest belangrijke features geselecteerd voor de machine learning model. Ik heb hierbij 2 technieken toegepast (voor feature selection):

o	Decision Tree
o	Lasso Regressie

Waaruit gebleken is dat Lasso Regressie niet de gewenste resultaten weergeeft. Decision tree wel, dit heb ik vervolgens als uitgangspunt genomen.

•	Feature Scaling, het scalen van alle features op de zelfde schaal

•	Ik heb mijn model getrained om de target variabele “deposit” te trainen. Hierbij heb ik een dimensie reductie techniek gebruikt genaamd PCA(Principle Component Analyse). Het model heb ik vervolgens getrained met meerdere componenten van PCA. Het voorspellen heb ik twee maal uitgevoerd. Eenmaal alleen voor de meest belangrijke features, en de tweede keer met alle features. Het blijkt uiteindelijk dat de beste resultaat wordt behaald door het trainen van alle features.


<details><summary><b><h3>Klik hier voor oude gemaakte code. Dit hoofdstuk is bewust niet verwijderd omdat het (wellicht niet al te goed volgens feedback) wel mijn aandeel laat zien binnen de groep. </h3></b></summary>
<p>
In dit hoofdstuk worden de stukjes code die ik heb gemaakt voor het project uitgelegd. Tevens worden er een aantal basis data science technieken beschreven die men acht te beheersen na deze minor.

&nbsp;

<h2>Predictive</h2>

Voor onze dataset bleek het beste dat er gekozen werd voor een supervised manier van leren. Hierbij zijn wij voorbarig begonnen met allerlei modellen te testen op onze data alvorens wij uberhaupt iets wisten over data science.
Na wat onderzoek en veel verder in het project, zijn wij tot conclusie gekomen dat de volgende 3 modellen het best werken voor onze dataset.

* Multinomial Naive Bayes
* Complement Naive Bayes
* Logistic Regression

Deze 3 predictive models hebben wij vervolgens ook toegepast op onze dataset. Daarbij behoort onder andere de volgende code:



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

Dit stukje code is het begin van het testen van ons model. Het bevat onder andere een preprocessor(voor het cleanen van de data). De code test op de 3 predictive modellen hierboven genoemd, ook split de code de dataset in een training / test / cross set.
Dit was in het begin niet voldoende, alleen testen op modellen bleek niet de juiste resultaat weer te geven op onze dataset. Na uitgebreid onderzoek en advies zijn wij terecht gekomen op zogeheten word embeddings.
Deze word embeddings geven tekst een bepaalde waarde in nummers. Hierdoor kan het model beter begrijpen welke woordcombinatie zwaarder weegt dan het ander. In dit geval zou het gebruikt worden om een bepaalde vraag te herkennen.
Zo hebben wij gekeken hoe een vraag zin wordt opgebouwd in het Nederlands en dit als input gegeven aan het model.

Omdat Word Embeddings meerdere modellen kent heb ik er twee gekozen om dit te gebruiken voor ons model. Dit waren de modellen TF - IDF Ngram en Count Vectors.

* TF - IDF 
* Count Vectors

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
<i>Predictive model met TFIDF - ngram op onze dataset</i>

Dit model traint op de aantal features en daarbij is gekozen voor een OneVsRest classifier.
Deze strategie zorgt ervoor dat je een classifier fit per class. Voor elke classifier is de class gefit tegen alle andere classes.
Het is gebruikelijk om voor de OneVsRest classifier te kiezen bij een Multi-class classification.
Tot slot split dit stukje code de data in Training, Test en een Cross Validation set.

Voor het model van de count_vectors is hetzelfde principe toegepast. Te zien hieronder:

* Count Vectors.
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
<i>Predictive model met count vectors op onze dataset.</i>

De toegevoegde waarde van mijn code op het project was het feit dat er ten eerste predictive model(len) zijn toegepast. Vervolgens zijn deze modellen ook met bovengenoemde vormen van word embedding toegepast op onze data.
Daaruit is gebleken welk model het beste was voor onze dataset. Uiteindelijk heeft het ervoor gezorgd dat er een vergelijking is gemaakt tussen allerlei modellen, en daaruit is het beste model gekozen.
Ook omdat elk model getest is tegenover Logistic Regression, Multinomial NB & Complement NB, kan de conclusie getrokken worden welk predictive model + word embeddings het best werken op de dataset.

Tot slot is gebleken dat de predictive model Logistic Regression met count vectors het beste model bleek te zijn voor ons dataset.


<i>Uiteindelijke resultaat van F1-score bij cross validatie set op datasets met verschillende verhoudingen van classificatie 3 vs de rest.</i>
![Test](/Portfolio/Courses/Screenshot%202019-01-11%20at%2020.06.46.png)


<h2>Preparation</h2>

Voor het project heb ik op het gebied van Data preparation een stukje datacleaning toegepast. 

Dit kan uitgevoerd worden in verschillende manieren, ik heb er twee toegepast op de dataset:

* <b> Aggregeren van Data </b> - Het in een leesbare tabel zetten van verkregen e-mail data(<i>uitgevoerd door mij op de dataset </i>)

Hiervoor heb ik gebruik gemaakt van de package pandas in Python. Dit is geleerd op de courses van datacamp:

```python
import pandas as pd


# Data laden van een .csv file
data = pd.DataFrame.from_csv('cbs.data')
# Converteren van data van een string naar tijd
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)
```

Dit was een stukje die ik had toegepast om in ieder geval de data van CBS e-mails in een datum formaat te zetten zodat het duidelijk was welke email op welke datum is verstuurd.

&


* <b> Het invoeren van data waar cellen leeg staan </b> - Meestal wordt er door een script gekeken waar data leegstaat.
Alle data die het model als <1 herkent wordt vervangen met een 0. Soms wordt het vervangen door een NaN = Not a Number.

Bij het zogeheten cleanen en voorbereiden van data heb ik een aantal tutorials gevolgd waarbij naar voren kwam hoe men leegstaande cellen zo goed mogelijk kon aanpakken. 
Daarbij is door mij de volgende code gehanteerd:

```python
# Lijst van alle leegstaande waarde
lege_waarden = ["n/a", "na", "--"]
df = pd.read_csv("cbs.csv", na_values = lege_waarden)
```

Zoals in de comment staat, zorgt dit stukje code ervoor dat het een lijst maakt van de dataset waarbij alle data die leegstaat wordt geinventariseerd.
Vervolgens heb ik gekozen om leegstaande vakken te vervangen door een nummer met de volgende code:

```python
# Leegstaande waarde veranderen door een nummer
df['cbs].fillna(125, inplace=True)
```


De toegevoegde waarde van het cleanen van data is dat er vooral een overzicht komt van de bruikbare data. Ook kan het ervoor zorgen dat men een verborgen patroon herkent in de data die voorheen niet gezien kon worden.
Tot slot kan het ervoor zorgen dat men bij het debuggen uren werk minder hoeft te doen wanneer je een diagnose / analyse gaat uitvoeren op je data.
Echter, moet er opgemerkt worden dat bij het teveel "cleanen" van data je veel cruciale data kan verliezen, gelukkig was mijn bijdrage niet te ingrijpend om dit daadwerkelijk zien te gebeuren.


Tevens heb ik samen met mijn collega's data gelabeld in 4 classificaties voor het predictive model. De code hiervoor en mijn aandeel staat beschreven in het kopje Predictive Models




<h2>Visualization</h2>

Confusion Matrix Visualisatie 

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

Het idee van het uitvoeren van een confusion matrix op de test set is om te kijken hoe het model een voorspelling uitvoert ten opzichte van onze classificatie.
Zo kon er vervolgens bepaald worden op welke punten het model niet een goede voorspelling deed, of dat er sprake was van een overfit of iets dergelijks.
Ook kon er een diagnose worden gedaan van de machine learning model in vorm van een Error Analyse.

In de gemaakte code kan men 2 soorten modellen van de confusion matrix onderscheiden:

* Normalized Confusion Matrix
* Non - Normalized Confusion Matrix


![test1](/Portfolio/Courses/Test-Normalized.png)
![test2](/Portfolio/Courses/Test-Not-Normalized.png)


* Van beide confusion matrix visualisaties is de input het geschetste model beschreven in hoofdstuk predictive modelling. Daar is ook beschreven dat Logistic Regression uiteindelijk het best mogelijke model is voor onze dataset.
* Te zien is hoe wij data hebben gelabeld op de "True Label" en hoe het model dit heeft voorspelt op de "Predicted Label".
* Alhoewel ik de confusion matrix op de Training, Cross validation en Test set heb gemaakt, schrijft de error analyse voor dat het uitgevoerd zal moeten worden op de test set. Vandaar de visualisatie van de test set.
* Normalization is gedaan om de snelheid van data te optimaliseren. Ook geeft het model verschil aan tussen zinnen en percentage. Dit bleek achteraf handig te zijn om letterlijk uit te vinden waarom een model iets voorspelt wat wij anders hebben geclassificeerd.

Credits: Om het bovenstaande te visualiseren heb ik samengewerkt met Timo Frionnet om de code te realiseren. 
Voor de error analyse gebaseerd op de gemaakte confusion matrix verwijs ik naar het kopje Evaluatie



<h2>Collection</h2>
Dit is niet relevant geweest voor ons onderzoek, aangezien alle beschikbare data door het CBS is vrijgegeven. Wellicht kan het stukje labelen vallen onder het kopje "Data Collection".
Hierbij hebben mijn collega's en ik de relevante datasets doorlopen en gelabeld als een 1,2,3 of 4 classificatie.


```python
Multi-class classification | Vier classificaties
De multi-class classification bestaat uit de volgende vier classificaties:
•	Niet-relevante beschikbaarheidsvraag: classificatie 1
Beschikbaarheidsvragen zijn niet-relevante vragen waarin een verzoek wordt gediend om informatie te verkrijgen over:
o	wanneer nieuwe cijfers gepubliceerd en/of geüpdatet zullen worden.
•	Niet-relevante verduidelijkingsvraag: classificatie 2
Verduidelijkingsvragen zijn niet-relevante vragen waarin een verzoek wordt gediend om informatie te verkrijgen over één of meerdere onderdelen:
o	totstandkoming van specifieke cijfers in publicaties en/of Statline.
o	definities van de gehanteerde termen in publicaties en/of Statline.
•	Niet-relevante zinnen: classificatie 3
Niet-relevante zinnen zijn zinnen waarin meningsuitingen over maatschappelijke onderwerpen worden gegeven, afsluitingen van e-mails en introductie van e-mails.

•	Relevante query vraag: classificatie 4
Query vragen zijn relevante vragen waarin een verzoek wordt gediend om informatie te verkrijgen over één of meerdere onderdelen:
o	specifieke cijfers van een onderwerp wat binnen een thema van het CBS valt. 
o	concrete gegevens van een onderwerp wat binnen een thema van het CBS valt. Denk hierbij aan inkomen wat onder CBS-categorie ‘Beroepsbevolking’ valt.
Bij een query vraag wordt de context van de vraag tevens als query vraag geclassificeerd, zodat de input van een query als volledig wordt beschouwd. 
```




<h2>Evaluation</h2>

```python

    def count_vectors(self, cvalue):
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_df=1.0, max_features=1500)
        count_vect.fit(self.trainDF['cleaned_sentence'])
        xtrain_count = count_vect.transform(self.X_train)
        xvalid_count = count_vect.transform(self.X_test)
        xcross_count = count_vect.transform(self.X_cross)

        for model_name, model in self.models.items():
            model.C = cvalue
            mc_model = multiclass.OneVsRestClassifier(model)
            classifier = mc_model.fit(xtrain_count, self.y_train)

            # Training predictions
            self.check_model(classifier, xtrain_count, self.y_train, model_name, features, 'count_vectors', 'training')

            # Test predictions
            self.check_model(classifier, xvalid_count, self.y_test, model_name, features, 'count_vectors', 'test')

            # Cross Validation predictions
            self.check_model(classifier, xcross_count, self.y_cross, model_name, features, 'count_vectors', 'cross')
```

Voor de evaluatie van het model om te zien wanneer er precies sprake is van een overfit of underfit is ongeveer hetzelfde soort code toegepast bij predictive modelling.
Echter, zijn er een aantal wijzigingen gemaakt:


* Data die uit mijn code is voortgevloeid is als .csv bestand in excel geimporteerd en vervolgens gevisualiseerd. Hierbij is geen code gebruikt van Python
* Input van data voor de visualisatie is de data output waarbij modellen die ik heb beschreven zijn gebruikt in het hoofdstuk: Predictive Modelling
* In plaats van features wordt nu gebruik gemaakt van cvalue (nieuwe feature)
* Max aantal features is nu 1500, uit eerdere tests van mijn collega's bleek dat na 1500 features sprake was van een overfit.
* Gehanteerde range is nu veranderd van -5 tot 5. 
* Alle gehanteerde inputs zijn de standaard inputs zoals max_df (verwijderen van termen dat te vaak voorkomen) = 1.0 etc. Deze bleek ook goed op de data te passen, te zien op de screenshot hier onder.


```python
    def get_and_print_all_scores(self):
        print('Running for count_vectors')
        for i in range(-5, 5):
            self.count_vectors(10**i)
            self.tfidf_words(i)
            self.tfidf_ngram(i)
            self.tfidf_char(i)
```
<i> Code voor het veranderen van de range </i>


![LG](/Portfolio/Courses/Screenshot%202019-01-11%20at%2013.53.33.png)
 
In dit model is te zien dat alles onder  <1 Range underfit is en boven de 1 Range overfit. Hierbij is te zien dat de default input van het model werkt op ons model.
Door middel van deze code is dus af te leiden voor de onderzoekers wanneer er sprake zou kunnen zijn van een overfit. 


Ook hebben wij constant in het project geevalueerd op F1 Score, Recall en Precision. Door deze scores te hanteren kon men in een opslag zien welk model het beste werkte voor onze dataset.

Aanvullend op de confusion matrix visualisatie, heb ik een diagnose uitgevoerd op dit model en de test dataset. Dit zorgde ervoor dat wij konden zien waar het model problemen had met het voorspellen.
Door middel van de error analyse herkende wij ook patronen in onze data, dit zorgde er direct voor dat wij het model zodanig konden tweaken dat het beter zou performen.

Een aantal punten die uit de error analyse is voortgekomen:

* Het model voorspelt vragen met aanhef meestal als geen vraag
* Het model heeft relevante vragen beter voorspelt dan de onderzoekers
* Het model had een precision & recall van bijna 80&



<h2>Diagnostics</h2>

</p>
</details>





<h1>Communicatie</h1>
    In dit hoofdstuk wordt er ingegaan op de presentaties, de research paper, taken in scrum & een korte reflectie over dit minor blok.

 <details><summary><h3>
    Klik hier voor inhoud van hoofdstuk Communicatie</h3></summary>
        <p>
    
<h2>Presentaties</h2>

Presentaties zijn altijd gezamenlijk gemaakt met de inbreng van de groep. Omdat ik, zoals aangegeven, meer in de kant van het onderzoeken was, presenteerde ik ook veel meer. 
In totaal heb ik 10 keer gepresenteerd waarvan 2 keer alleen. 

Presentaties per week:

* [Week 1](/Presentaties/2018.08.31-intro.pptx)
* [Week 2](/Presentaties/2018.09.07%20Presentatie.pptx) 
* [Week 3](/Presentaties/2018.09.10%20CBS%20Presentatie.pptx)
* [Week 4](/Presentaties/2018.09.14%20Presentatie.pptx)
* [Week 5](/Presentaties/2018.09.21%20CBS%20Presentatie.pptx)
* [Week 6](/Presentaties/2018.09.28%20CBS%20Presentatie.pptx)
* [Week 7](/Presentaties/2018.10.05%20CBS%20Presentatie.pptx)
* [Week 8](/Presentaties/2018.10.12%20CBS%20Presentatie.pptx)
* [Week 9](/Presentaties/2018.10.19%20CBS%20Presentatie.pptx)
* [Week 10](/Presentaties/2018.11.02%20CBS%20Presentatie.pptx)
* [Week 11](/Presentaties/2018.11.09%20CBS%20Presentatie.pptx)
* [Week 12](/Presentaties/2018.11.16%20CBS%20Presentatie.pptx)
* [Week 13](/Presentaties/2018.11.30%20CBS%20Presentatie.pptx)
* [Week 14](/Presentaties/2018.12.07%20CBS%20Presentatie.pptx)
* [Week 15](/Presentaties/2018.12.17%20CBS%20Presentatie%20%5BAutosaved%5D.pptx)
* [Week 16](/Presentaties/2018.12.21%20CBS%20Presentatie.pptx)


<h2>Research Paper</h2>
   
De paper is een gezamenlijke bijdrage van de gehele groep. 
Omdat ik geen fervente coder ben, heb ik samen met mijn collega Seyma Irilmazbilek vooral gericht tot de taak onderzoeken en delen van kennis(o.a. aanpak, ideeen etc.) aan ons groepsgenoten binnen dit blok. 
Zodoende was de paper meer mijn domein. 

Zo heb ik het volgende uitgevoerd binnen de paper:

* Related Work
* Gedeelte Methode
* Gedeelte Aanpak
* Bronnen uitzoeken relevant voor ons opdracht, uitdragen en citeren in het verslag (Graag verwijs ik naar kopje literatuur)
* (Code)Error Analyse samen met Timo Frionnet 
* Conclusie & Discussie
* APA Verslag, Figuren, Vergelijkingen, Bronnen
* Layout

Het gehele rapport zal ook apart worden ingeleverd. I.v.m. de vertrouwelijkheid van data zal ik niet naar het rapport verwijzen in dit portfolio.


<h2>Scrum</h2>
Omdat ik niet de grootste fan van scrum ben, heb ik dit ook nauwelijks gebruikt. Voor een breakdown van Scrum per persoon, refereer ik naar de Scrumwise: https://www.scrumwise.com/scrum/#/people/project/kb74-2018-cbs pagina van de CBS projectgroep.

Taken en activiteiten beknopt:  

* Domein Studie: Dit houdt in dat ik onderzoek deed naar alle gerelateerde onderzoeken op dit gebied. Maar ook naar bepaalde gehanteerde methodieken etc. Dit heb ik vervolgens geinventariseerd en uitgedragen binnen het project groep.
* Het onderzoeksplan voor ons opdracht opstellen en presenteren aan het CBS
* Linear Classifier Methode Onderzoeken & Beschrijven
* Logistic Regression Methode Onderzoeken & Beschrijven
* Word Level TF - IDF Methode Onderzoeken & Beschrijven
* Linear Regression & TF - IDF & Naive Bayes Voordelen, Nadelen omschrijven per type model
* Language Models Onderzoeken
* Formule uitleggen Extreme Gradient Boosting & Linear Regression
* Onderzoek Deep Learning Models
* Related Work - Bronnen onderzoeken gerelateerd aan ons project
* Research Paper
* POS Tagging > Text Classificatie methoden en modellen onderzoek
* Multinomial Naive Bayes > Onderzoek
* Text Classificatie methoden en modellen onderzoeken & uitschrijven
* Datasets labelen in 4 classificaties
* Confusion Matrixes maken van gelabelde datasets
* Error Analyse over Training - Test - Cross set



<h2>Reflectie</h2>
Het waren hele leerzame 2 blokken en voorheen had ik nooit gedacht dat ik mij zou verdiepen in dit onderwerp. Wat ben ik enerzijds blij dat ik het toch heb gedaan maar anderzijds ook weer niet.
Dit blok kent veel up's en down's voor mij zowel op persoonlijk vlak als op het "zakelijke". Zo kon ik niet altijd opbrengen om aan onderwerpen te zitten wat ik veelal niet begreep. 
Wellicht kwam dit omdat mij interesse er ook niet naar was. Dit neem ik mijzelf kwalijk. Ook vind ik dat ik sommige onderdelen van de course te laks heb aangepakt. Ik had meer inzet moeten tonen en meer moeten willen.
Ik vind wel dat ik mijzelf heb herpakt, en dat ook heb laten zien de afgelopen weken, maar die laksheid is toch iets wat in mijn aard zit af en toe. Dan gooi ik er met de pet naar.  

Daarentegen waren er onderwerpen die ik probleemloos afmaakte, ook het coderen van python en het keer op keer falen vond ik niet erg. Ik had er mijn passie in gevonden en ik zou het blijven doen tot ik er voldoende skilled in was.
Ik heb bovendien heel veel geleerd op het gebied van Machine Learning, vooral het onderzoeken en het vertalen naar een model was leerzaam. Ik heb wel het idee dat ik aardig kan meepraten als men het nu over data science heeft.
Wellicht was dat mijn doel ook al die tijd, ik had sowieso niet verwacht de beste data scientist te worden binnen 5 maanden. Stiekem had ik wel verwacht dat ik een carriere in de data science zag.
Dat is nu wel veranderd, ik denk dat ik toch meer hou van mijn eigen vakgebied (IT-Security).

Sommige lessen gingen wel eens van 0 naar 100 voor mijn gevoel, lang leve de internet en jaar 2018 waar alles tegenwoordig online te vinden staat zeg ik dan maar. Ik heb ook een hele goede groep gehad moet ik bekennen die elkaar steunde door dik en dun.
Wij als groep hebben het ook niet makkelijk gehad, zo zijn er vroegtijdig al 2 groepsgenoten ons project verlaten. Dit zorgde ervoor dat ons groep tijdelijk ons evenwicht kwijtraakte. Gelukkig hebben wij dat kunnen herpakken.

Ik heb veel geleerd over Data Science en hoe men dit kan toepassen in de praktijk. Maar vooral van het vak zelf. Wat inspireert mensen, wat beweegt mensen. Hoe krijg je van niets naar een prachtig model. Het is toch een passie die niet voor vele is weggelegd. Maar als men mij zal vragen zal je ooit iets met data science doen? Wellicht, maar niet in de nabije toekomst.
Wel zal ik Python vaak gebruiken, dat is een taal die ik inmiddels goed heb omarmd en dit zal ik dan ook zeer zeker voor een lange tijd gebruiken. 

Ik vind wel dat de course een strengere eis moet gaan stellen voor het programmeren en het begrijpen van wiskunde op gebied van statistiek. Ik begrijp overigens dat de minor niet vraagt dat je opeens een prodigy bent na 5 maanden, maar toch is het soms lastig in te komen.

Tot slot wil ik zeggen dat ik wel een positief gevoel heb over gehouden aan de leraren en de sfeer in de klas. De leraren wisten waar ze over praatte en bij Jeroen had ik het idee dat er geen een onderwerp was in Machine Learning waar hij geen weet van had.
Dat is toch prettig af en toe, dat je leraren hebt die weten waar ze over praten. Dat is naar mijn ervaring niet altijd zo geweest.
</p></details>
