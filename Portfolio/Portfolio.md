<h1>Persoonlijke Portfolio</h1>

Het persoonlijke portfolio voor de minor Data Science aan de Haagse Hogeschool

* <b>Naam</b>: Askin Sarikaya
* <b>Studentnummer</b>: 14121409
* <b>E-mail</b>: 14121409@student.hhs.nl

<h2>Introductie</h2>

This is a general introduction to the KB74-OPSCHALER project and it is also intended for people who do not know anything about the Applied Data Science minor.

In the OPSCHALER project multiple universities and (energy) companies collaborate in developing methods and tools to extract useful information out of the energy usage data of residential houses on different aggregation levels. For more information on OPSCHALER itself, see their website: www.opschaler.nl.

The Hague University is one of the collaborators and offers the OPSCHALER data to students in their Applied Data Science minor, which takes one semester (30 ECT). This leads us to the KB74-OPSCHALER project.

 Click here for information regarding the KB74-OPSCHALER team.
Our team consists of 6 students, 2 of which are doing this for their European Project Semester. Every one of us has different backgrounds. One person has a BSc in telecommunication engineering and is currently doing a masters. The others are studying industrial engineering and management, computer science engineering and engineering physics. Another important note is that most of us have never programmer before, or only in MATLAB. So along with learning the subjects from the minor, most of us had to learn Python from scratch aswell.

Our research started out by trying to predict the electricity and gas consumption of individual residential houses on a 10 second and one-hour resolution, respectively. The time we wanted to predict ahead was one hour to a week, by using as less data as possible. Due to model complexity, time, and scarce messy data, the research got narrowed down to predicting the gas consumption of houses on the aggregated level, predicting one hour, a day and a week ahead with an hourly resolution. This is done by only using historical and future weather information. Whereas the aggregated level in our case consists of the mean gas usage of 54 houses and could represent a block of houses. These predictions are done by using the different models listed below and are eventually compared to each other.

MVLR: Multivariate Linear Regression
DNN: Deep Neural Network
CNN: Convolutional Neural Network
RNN: Recurrent Neural Network
LSTM: Long Short-Term Memory
GRU: Gated Recurrent Unit
TimeDistributed(CNN)+RNN+DNN
Despite these models being based on data on the aggregated level, they should also work for individual houses when trained specifically for that house. Creating an accurate general model, using only weather data is just hard due to each house having a specific gas consumption pattern.


<h1>Domain Knowledge</h1>
<h2>Jargon</h2>
 TODO: Add more jargon. 
Used jargon for Opschaler is listed below.

Dwelling: an unique house.
Smartmeter data: electricity and gas meter data.
gasPower: amount of gas being used at a given time.
ePower: amount of electricity being used at a given time.
smart: electricity DataFrame of a dwelling.
seq2seq: sequence to sequence
 (add screenshots of the online courses you have finished (DataCamp, Coursera, etc))
Domain Knowledge (Literature, jargon, evaluation, existing data sets, ...)
<h2>Literature</h2>
<h1>Courses</h1>
Include information of the following subjects:
<h2>Datacamp Courses </h2>
<h2>Coursera Courses </h2>
<h2>Python Notebooks

<h1>Data & Modellen</h1>
<h2>Predictive Models</h2>
Hier stuk over supervised learning, classification of regression
<h2>Data preparation</h2>
Stuk over labelen 
<h2>Data Visualization</h2>
Stuk over confusion matrix
<h2>Data collection</h3>
Stuk over labelen
<h2>Evaluation</h3>
Zelf iets maken
<h2>Diagnostics of the learning process</h2>

<h1>Communicatie</h1>
<h2>Presentaties</h2>
<h2>Paper</h2>
Bijdrage Paper beschrijving

<h2>Scrum</h2>
(presentations, summaries, paper, ...)
Link to the Python Notebooks you have finished (you can dump them to PDF)
List the tickets from the Scrum backlog that you worked on, linked to deliverables, own experiments, etc.
Add any other assisdadsasdgnment you feel is evidence of your abilities