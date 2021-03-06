# Identifying-Ham-Or-Spam-Email-Using-Text-Analytics---R-Programming
Identifying Ham Or Spam Email Using Text-Analytics R-Programming

Problem Statement :-
---------------------

Every email user has experienced spam emails which may contain unsolicited messages or harmful virus.
They can attack a users privacy and is one of the cyber-security concern.

The emails can be divided in to two parts Spam and other emails known as Ham.

The aim of this project is to identify which emails are Ham or Spam using text analytics and to
build a predictive model in which we will be extracting word frequencies from the text of the documents, 
and then integrating those frequencies into predictive models.

The main focus of this project is to showcase 'predictive coding' – an emerging use of text analytics in the area of criminal justice.
============================================================================================================================================================================================

The Problem description
------------------------
we will build and evaluate a spam filter using a publicly available dataset first described in the 2006 conference paper “Spam Filtering with Naive Bayes – Which Naive Bayes?”
by V. Metsis, I. Androutsopoulos, and G. Paliouras. The “ham” messages in this dataset come from the inbox of former Enron Managing Director for Research Vincent Kaminski, 
one of the inboxes in the Enron Corpus. One source of spam messages in this dataset is the SpamAssassin corpus, which contains hand-labeled spam messages contributed by Internet users. 
The remaining spam was collected by Project Honey Pot, a project that collects spam messages and identifies spammers by publishing email address that humans would know not to contact
but that bots might target with spam.

============================================================================================================================================================================================

Predictive Coding
------------------

Predictive coding is a new technique in which attorneys manually label some documents and then use text analytics models trained on the manually labeled documents
to predict which of the remaining documents are responsive.

============================================================================================================================================================================================

DATA SET
----------
The data set contains just two fields:

a.) text: the text of the email in question,
b.) spam: a binary (0/1) variable telling whether the email was spam.

=============================================================================================================================================================================================

Creating the corpus
-------------------
Follow the standard steps to build and pre-process the corpus:

1) Build a new corpus variable called corpus.

2) Using tm_map, convert the text to lowercase.

3) Using tm_map, remove all punctuation from the corpus.

4) Using tm_map, remove all English stopwords from the corpus.

5) Using tm_map, stem the words in the corpus.

6) Build a document term matrix from the corpus, called dtm.

we are calling our corpus as the dtm

=======================================**********************************************=======================================================================================================
About The Corpus :- 
----------------------
In linguistics, a corpus (plural corpora) or text corpus is a large and structured set of texts (nowadays usually electronically stored and processed).
They are used to do statistical analysis and hypothesis testing, checking occurrences or validating linguistic rules within a specific language territory.

Text mining and certain plotting packages are not installed by default so one has to install them manually 

The relevant packages are:

tm – the text mining package. 
SnowballC – required for stemming.
ggplot2 – plotting capabilities
wordcloud – which is self-explanatory


#########################################################################################################################################################################################
In text mining, why should we remove the sparse term from the document term matrix?

In text mining, when you use a bag of words approach,  ignoring terms that have a document frequency lower than a given threshold can help generalization and prevent overfitting.
Think about it this way: let's say you have a large corpus of documents. You'll have some words that show up, let's say, only once. 
Since they show up only once they are either always associated to one class or the other (in a binary classification problem). You then sit there and wonder whether,
if you had more document with that word, you would still observe that strong association, or if the association wouldn't generalize. You can generalize that idea,
and ask the same question for thresholds of 2, 5, 10...
Essentially you're looking for a minimum threshold that improves generalization of your model. This is where, eliminating words with low frequency, usually helps.
You should consider this threshold a hyperparameter, and try different settings. I usually try 1, 2, 3, 5, 10, 20 and get a feeling for how performance changes.

##########################################################################################################################################################################################
Sort The Words
---------------
colSums() is an R function that returns the sum of values for each variable in our data frame.
Our data frame contains the number of times each word stem (columns) appeared in each email (rows). 
Therefore, colSums(emailsSparse) returns the number of times a word stem appeared across all the emails in the dataset. 
What is the word stem that shows up most frequently across all the emails in the dataset? 
Hint: think about how you can use sort() or which.max() to pick out the maximum frequency.

















