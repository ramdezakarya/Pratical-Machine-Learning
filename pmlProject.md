---
title: "pml_project"
author: "RAMDE ZAKAYA"
date: "Thursday, June 4, 2015"
output: html_document
---

###Introduction

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

Before beginning, we have to get on some libraries necessary for the project :

```{r}
library(rpart)
library(rpart.plot)
library(rattle)
library(caret)
library(randomForest)

```


###Data loading 

```{r}
setwd("C:/Users/zakarya.ramde/Desktop/Coursera/Data Scientist/Pratical Machine Learning")
training = read.csv("pml-training.csv", header=TRUE, na.strings=c("NA","DIV/0!", ""))
Testing = read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA","DIC/0!",""))
dim(training); dim(Testing) 
```
As we can see, the training set has 19 622 rows for 160 variables. 

###Data cleaning 

Explore data, we see many columns with more NA values or empty cells. So, we have to clean data by delete this columns. We begin with first colum (ID column). 
Analyzing with the view table, column, we see that seven first columns haven't some importance for the prediction so we can delete them also. 


###Data partition 

For data partition, we devided the training data into 60% training and 40% testing.

```{r}
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training <- training[inTrain,]
testing <- training[-inTrain,]

```


##Data cleaning 

Seeing the training database, we see many cells with NA values or DIV/0. We have to clean it by suppressing of these variables and consider only those which have correct data.
First, we delete the first seven columns in relation with time variable and X. 

```{r}
training <-training[,-c(1:7)]
testing <- testing[,-c(1:7)]
Testing <- Testing[,-c(1:7)]
NAS <-as.vector(sapply(training[,1:153], function(x) {length(which(is.na(x)))!=0}))
training <- training[,!NAS]
testing <- testing[,!NAS]
Testing <- Testing[,!NAS]
```



##Model construction 

Firstly, we use the rpart method on training data to see if it gives good results about prediction.

```{r}
set.seed(2)
modFit <- train(classe ~., method="rpart", data=training)
modFit
fancyRpartPlot(modFit$finalModel)
predictS <-predict(modFit, testing)
confusionMatrix(predictS,testing$classe)
```


The results show that, we have about 50% of accuracy for this model. These results are not good for prediction. So for the next step, we will use the boosting method (gbm).


```{r}
set.seed(2)
#we define 10 fold control for cross validation 
contr <- trainControl(method="cv", number=10)
modFit1 <- train(classe ~., method="gbm", data=training, trControl=contr, verbose=FALSE)
modFit1
```


```{r}
predictS1 <-predict(modFit1, testing)
confusionMatrix(predictS1,testing$classe)
```

The results show that we get an accuracy of 97% with a CI interval of 96.77 and 97.72. The sensitivity is about 99% and also the specificity. 
So the gbm method give the good model for prediction. 


##Submission 

```{r}
answers <- predict(modFit1, Testing)

```

```{r}
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(answers)

```








