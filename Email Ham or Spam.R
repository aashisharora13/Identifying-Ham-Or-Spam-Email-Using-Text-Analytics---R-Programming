Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")

# Read Data Files - emails.csv
emails <- read.csv('emails.csv', stringsAsFactors = FALSE)

# Knowing structure of the emails file
str(emails)
head(emails)

# Let's look at a few examples (using the strwrap() function for easier-to-read formatting):
## First email
strwrap(emails$text[1])

## Second Email
strwrap(emails$text[2])

emails$spam[2]

## word appearing at the beginning of every email in the dataset
emails$text[1]

## How many characters are in the longest email in the dataset 
## where longest is measured in terms of the maximum number of characters?
max(nchar(emails$text)) #-- 43952

## print the longest email present in the dataset
which(nchar(emails$text)== 43952) #-- 2651
strwrap(emails[2651,])

## Smallest email measured in terms of minimum number of characters?
min(nchar(emails$text)) #-- 13
which(nchar(emails$text) == 13) #== Row Number = 1992
strwrap(emails[1992,])


##--------------------------------------------------------------------------##
#  breakdown of the number of emails which are spam and not spam.
table(emails$spam) #-- 0 => 4360 and 1 => 1368
# We see that the data set is unbalanced, with a relatively small proportion of emails 
# responsive to the query. 
# This is typical in predictive coding problems.
## --------------------------------------------------------------------------##

## CREATING A CORPUS ##
library("tm")
corpus <- Corpus(VectorSource(emails$text))

## preprocessing the corpus ## 
#  tm_map() function which takes as
# a.) its first argument the name of a corpus and
# b.) second argument a function performing the transformation that we want to apply to the text.

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, PlainTextDocument)
corpus <- tm_map(corpus, removePunctuation)

## Removing Stop Words ##
# Removing words can be done with the removeWords argument to the tm_map() function, 
# with an extra argument, i.e. what the stop words are that we want to remove, 
# for which we simply use the list for english that is provided by the tm package.
# We will remove all of these English stop words, but we will also remove the word "apple" 
# since all of these tweets have the word "apple" and it probably won't be very useful in 
# our prediction problem.

corpus <- tm_map(corpus,removeWords, stopwords("english"))

## Stemming ##

# stem our document with the stemDocument argument. #

corpus <- tm_map(corpus, stemDocument)

## the emails in this corpus are ready for our machine learning algorithms. ##

# BAG OF WORDS #
## Creating Document Term Matrix
### extract the word frequencies to be used in our prediction problem.
#### The tm package provides a function called DocumentTermMatrix() that
#### generates a matrix where :
#### a.) the rows correspond to documents, and
#### b.) the columns correspond to words.

#### The values in the matrix are the number of times that word appears in each document.
dtm <- DocumentTermMatrix(corpus)

#### To obtain a more reasonable number of terms, limit dtm to contain terms appearing 
#### in at least 5% of documents, and store this result as spdtm,
#### and view the number of terms present in the spdtm?
#### We want to remove the terms that don't appear too often in our data set.

# Remove sparse terms
spdtm <- removeSparseTerms(dtm,0.95)

## We can see that we have decreased the number of terms to 330, 
## which is a much more reasonable number.

# Creating a data Frame from the DTM
## data frame called emailsSparse from spdtm, and use the make.names 
## function to make the variable names of emailsSparse valid
emailsSparse <- as.data.frame(as.matrix(spdtm))

# use the make.names function to make the variable names of emailsSparse valid
colnames(emailsSparse) <- make.names(colnames(emailsSparse))

# Sort The Words
## What is the word stem that shows up most frequently across all the emails in the dataset?
sort(colSums(emailsSparse))

which.max(colSums(emailsSparse))

# Adding the variable
## Add a variable called "spam" to emailsSparse containing the email spam labels,
## this can be done by copying over the "spam" variable from the original data frame.

emailsSparse$spam = emails$spam

## Now let's see how many time word stems appear at least 5000 times in the ham emails 
## in the dataset We can read the most frequent terms in the ham dataset

sort(colSums(subset(emailsSparse, spam == 0)))

# SUbset the spam emails
spamEmails <- subset(emailsSparse, spam == 1)

# Reading Most Frequent Terms
sort(colSums(spamEmails))

#** Building a Machine Learning Model **#
## First, convert the dependent variable to a factor
## with emailsSparse$spam = as.factor(emailsSparse$spam)
emailsSparse$spam <- as.factor(emailsSparse$spam)

## we are setting the random seed some value so that every time same result will come.
set.seed(123)

## before building the model we need to split our data into training and testing 
## by using sample.split function with 70 data in training and rest in test

spl <- sample.split(emailsSparse$spam, 0.7)

## Train and Test Subset
## use the subset function TRUE for the train and FALSE for the test

train <- subset(emailsSparse, spl == TRUE)
test <- subset(emailsSparse, spl == FALSE)

# Logistic Regression Model
## we are creating a logistic regression model called as spamLog, 
## for logistic regression we use function glm with family binomial. 
## we are using the all the variable as independent variable for training our model, for detection.

spamLog <- glm(spam~.,data = train, family = 'binomial')

## Summary of Logistic Regression Model
summary(spamLog)

# Build a CART Model
## create one model CART and called this model as spamCART, we are using default parameters 
## to train this model, so that no need to add minbucket or cp parameters. Remember to add
## the argument method="class" since this is a binary classification problem.

spamCART <- rpart(spam~., data = train, method = 'class')

# Plot CART
plot(spamCART)
printcp(spamCART)
plotcp(spamCART)
summary(spamCART)
prp(spamCART)

library(rattle)

# BUILD A RANDOM FOREST
## We are creating one more model random forest model and called this model as spamRF, 
## we are using default parameters to train this mode also, no need to worry about specifying
## ntree or nodsize. Directly before training the random forest model, set the random seed to 123 (even though we've already done this, it's important to set the seed right before training the model so we all obtain the same results. Keep in mind though that on certain operating systems, your results might still be slightly different).
install.packages("randomForest")
set.seed(123)
spamRF <- randomForest(spam~., data=train)

# Out-of-Sample Performance of all the above model we created
predTrainLog <- predict(spamLog, type="response")
predTrainCART <- predict(spamCART)[,2]
predTrainRF <- predict(spamRF,type = "prob")[,2]

#This new object gives us the predicted probabilities on the test set.
# Predicted Probabilities
# 1.) How many of the training set predicted probabilities from spamLog are less than 0.00001?
table(predTrainLog<0.00001)
# 2.) How many of the training set predicted probabilities from spamLog are more than 0.99999?
table(predTrainLog>0.99999)
# 3.) How many of the training set predicted probabilities from spamLog are between 0.00001 and 0.99999?
table(predTrainLog >= 0.00001 & predTrainLog <= 0.99999 )

summary(spamLog)

## Accuracy
# What is the training set accuracy of spamLog, using a threshold of 0.5 for predictions.
table(train$spam, predTrainLog > 0.5)

(3052+954)/nrow(train) # -- 0.9990025

# What is the training set accuracy of spamCART, using a threshold of 0.5 for predictions?
table(train$spam, predTrainCART > 0.5)
# Accuracy
(2885+894)/nrow(train) #-- 0.942394

# What is the training set accuracy of spamRF, using a threshold of 0.5 for predictions?
table(train$spam, predTrainRF > 0.5)
# Accuracy
(3013+914)/nrow(train) #-- 0.9793017

#** EVALUATING ON THE TEST SET **#
# We are interested in the accuracy of our models on the test set, i.e. out-of-sample.
# First we compute the confusion matrix:
predTestlog <- predict(spamLog, newdata = test, type = "response")
predTestCART <- predict(spamCART, newdata = test)[,2]
predTestRF <- predict(spamRF, newdata = test, type = "prob")[,2]

# Test Accuracy of all the above model
## testing set accuracy of spamLog, using a threshold of 0.5 for predictions?
table(test$spam, predTestlog > 0.5)
## Accuracy of Logistic Regression Model
(1257+376)/nrow(test) #-- 0.95

table(test$spam, predTestCART > 0.5)
## Accuracy of CART Model
(1228+386)/nrow(test) #-- 0.93

table(test$spam, predTestRF > 0.5)
## Accuracy of Random Forest
(1290+385)/nrow(test) #--0.97




