---
title: "pml_project"
author: "RAMDE ZAKAYA"
date: "Thursday, June 4, 2015"
output: html_document
---

##Introduction

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

Before beginning, we have to get on some libraries necessary for the project :


```r
library(rpart)
library(rpart.plot)
library(rattle)
```

```
## Rattle : une interface graphique gratuite pour l'exploration de données avec R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Entrez 'rattle()' pour secouer, faire vibrer, et faire défiler vos données.
```

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## 
## Attaching package: 'caret'
## 
## The following object is masked _by_ '.GlobalEnv':
## 
##     best
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


##Data loading 


```r
setwd("C:/Users/zakarya.ramde/Desktop/Coursera/Data Scientist/Pratical Machine Learning")
training = read.csv("pml-training.csv", header=TRUE, na.strings=c("NA","DIV/0!", ""))
Testing = read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA","DIV/0!",""))
dim(training); dim(Testing) 
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
As we can see, the training set has 19 622 rows for 160 variables. 


##Data partition 

For data partition, we devided the training data into 60% training and 40% testing.


```r
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training <- training[inTrain,]
testing <- training[-inTrain,]
```


##Data cleaning 

Seeing the training database, we see many cells with NA values or DIV/0. We have to clean it by suppressing of these variables and consider only those which have correct data.
First, we delete the first seven columns in relation with time variable and X. 


```r
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


```r
set.seed(2)
modFit <- train(classe ~., method="rpart", data=training)
modFit
```

```
## CART 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03409271  0.5208530  0.37975544  0.04367473   0.06955088
##   0.03808733  0.5015408  0.35286390  0.04932517   0.08066700
##   0.11366872  0.3349098  0.07776554  0.04073420   0.05960429
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03409271.
```

```r
fancyRpartPlot(modFit$finalModel)
```

![plot of chunk unnamed-chunk-5](assets/fig/unnamed-chunk-5-1.png) 

```r
predictS <-predict(modFit, testing)
confusionMatrix(predictS,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 811 157  27  35  15
##          B   4 162  17   3   3
##          C 382 430 482 229 333
##          D 112 192 285 507 130
##          E   1   0   0   0 406
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5014         
##                  95% CI : (0.487, 0.5157)
##     No Information Rate : 0.2774         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3802         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6191  0.17216   0.5943   0.6550  0.45772
## Specificity            0.9314  0.99286   0.6488   0.8179  0.99974
## Pos Pred Value         0.7761  0.85714   0.2597   0.4135  0.99754
## Neg Pred Value         0.8643  0.82819   0.8852   0.9236  0.88855
## Prevalence             0.2774  0.19924   0.1717   0.1639  0.18780
## Detection Rate         0.1717  0.03430   0.1021   0.1073  0.08596
## Detection Prevalence   0.2213  0.04002   0.3930   0.2596  0.08617
## Balanced Accuracy      0.7753  0.58251   0.6216   0.7365  0.72873
```


The results show that, we have about 50% of accuracy for this model. These results are not good for prediction. So for the next step, we will use the boosting method (gbm).



```r
set.seed(2)
#we define 10 fold control for cross validation 
contr <- trainControl(method="cv", number=10)
modFit1 <- train(classe ~., method="gbm", data=training, trControl=contr, verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
```

```r
modFit1
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 10597, 10597, 10600, 10599, 10598, 10598, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7485579  0.6812341  0.012633552
##   1                  100      0.8188658  0.7707789  0.013364582
##   1                  150      0.8517284  0.8124118  0.010294656
##   2                   50      0.8556385  0.8171184  0.009386885
##   2                  100      0.9048909  0.8796474  0.008516737
##   2                  150      0.9301969  0.9116769  0.006946863
##   3                   50      0.8949540  0.8670161  0.008586242
##   3                  100      0.9398774  0.9239262  0.006020234
##   3                  150      0.9583898  0.9473589  0.006402640
##   Kappa SD   
##   0.015884204
##   0.016861558
##   0.012977990
##   0.011900501
##   0.010774165
##   0.008784378
##   0.010923732
##   0.007605495
##   0.008107941
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```



```r
predictS1 <-predict(modFit1, testing)
confusionMatrix(predictS1,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1296   20    0    0    2
##          B   10  908   21    1    4
##          C    2   13  781   17    3
##          D    2    0    7  753    7
##          E    0    0    2    3  871
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9759        
##                  95% CI : (0.9711, 0.98)
##     No Information Rate : 0.2774        
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.9695        
##  Mcnemar's Test P-Value : 0.0142        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9893   0.9649   0.9630   0.9729   0.9820
## Specificity            0.9936   0.9905   0.9911   0.9959   0.9987
## Pos Pred Value         0.9833   0.9619   0.9571   0.9792   0.9943
## Neg Pred Value         0.9959   0.9913   0.9923   0.9947   0.9958
## Prevalence             0.2774   0.1992   0.1717   0.1639   0.1878
## Detection Rate         0.2744   0.1923   0.1654   0.1594   0.1844
## Detection Prevalence   0.2791   0.1999   0.1728   0.1628   0.1855
## Balanced Accuracy      0.9914   0.9777   0.9770   0.9844   0.9903
```

The results show that we get an accuracy of 97% with a CI interval of 96.77 and 97.72. The sensitivity is about 99% and also the specificity. 
So the gbm method give the good model for prediction. 


##Submission 


```r
answers <- predict(modFit1, Testing)
```


```r
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}
pml_write_files(answers)
```








