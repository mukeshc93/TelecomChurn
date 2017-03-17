# Telecom Churn 
This is a document containing sample starter scripts using various ML techniques and algorithms
---
title: '**TELECOM CHURN **'
output:
  html_notebook: default
  html_document: default
  pdf_document: default
  word_document: default
---

> ####**This report will list down all my approaches and outputs for the codes and algorithms tried throughout the project.**


###`Approach 1: Random Forest on Entire data`
```{r, message=FALSE, warning=FALSE}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

df$target=as.factor(df$target)

set.seed(134);x=sample(1:25000,0.8*25000)
train_data_num=df[x,]
test_data_num=df[-x,]

library(randomForest)
ml=randomForest(target~.,data=train_data_num)

p_r_train=predict(ml,train_data_num)
p_r_test <- predict(ml,test_data_num)
```  
  
> Confusion matrix on train data    

```{r, echo=FALSE}
table(Predicted=p_r_train,Actual=train_data_num$target)
```
> Confusion matrix on test data

```{r, echo=FALSE}
table(Predicted=p_r_test,Actual=test_data_num$target)  
```

** **  

> Using a custom function to find accuracy and other metrics

```{r}
metrics=function(actual,predicted)
{
  actual=as.numeric(as.character(actual));predicted=as.numeric(as.character(predicted))
  accuracy=sum(actual== predicted)/length(actual);print(paste("accuracy: ", accuracy))
  recall=sum(predicted & actual) / sum(actual);print(paste("recall: ",recall))
  p=sum(predicted & actual) / sum(predicted);print(paste("precision: ",p))  
  f=2*p*recall/(p+recall);print(paste("F1 Score: ",f))
}
```

**  **  
**`Metrics on train data`**  

```{r, echo=FALSE}
metrics(train_data_num$target,p_r_train)
```
**  **

**`Metrics on test data`** 
```{r, echo=FALSE}
metrics(test_data_num$target,p_r_test)
```

**  **
**  **

###`Approach 2: Logistic regression on Entire data`
```{r, message=FALSE, warning=FALSE}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

df$target=as.factor(df$target)

set.seed(134);x=sample(1:nrow(df),0.8*nrow(df))
train_data_num=df[x,]
test_data_num=df[-x,]

log_model=glm(target~.,data=train_data_num,family=binomial)
p_train <- predict(log_model,train_data_num, type="response")
p_r_train=(ifelse(p_train>0.3,1,0))

p_test <- predict(log_model,test_data_num, type="response")
p_r_test=(ifelse(p_test>0.3,1,0))

pr <- prediction(p_train, train_data_num$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf)
abline(a=0, b= 1)

```


> ###The AUC obtained is `r auc`

> Confusion matrix on train data

```{r, echo=FALSE}
table(Predicted=p_r_train,Actual=train_data_num$target)
```
> Confusion matrix on test data

```{r, echo=FALSE}
table(Actual=test_data_num$target,Predicted=p_r_test)
```
**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(train_data_num$target,p_r_train)

```

**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(test_data_num$target,p_r_test)
```


**  **
**  **

###`Approach 3: Logistic regression on PCA components`
Here we are using attributes with only continuous values, as the discrete values when normalized can cause issues while calculating the principal components.

```{r}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")
numr_attr=sapply(df,is.integer)
numeric_data=df[,!numr_attr]
set.seed(134);x=sample(1:nrow(df),0.8*nrow(df))
train_data_num=numeric_data[x,]
prin_comp=prcomp(train_data_num,scale. = T)

train_data=as.data.frame(prin_comp$x[,1:12])##Taking 12 PCA components as they explain close to 80% variance
train_data$target=as.factor(df$target[x])

test_data_num=numeric_data[-x,]
test_data=as.data.frame(predict(prin_comp,test_data_num)[,1:12])
test_data$target=as.factor(df$target[-x])

log_model=glm(target~.,data=train_data,family=binomial)

p_train <- predict(log_model,train_data, type="response")
p_r_train=(ifelse(p_train>0.3,1,0))

p_test <- predict(log_model,test_data, type="response")
p_r_test=(ifelse(p_test>0.3,1,0))

pr <- prediction(p_train, train_data$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf)
abline(a=0, b= 1)

```


> ###The AUC obtained is `r auc`

> Confusion matrix on train data

```{r, echo=FALSE}
table(Predicted=p_r_train,Actual=train_data$target)
```
> Confusion matrix on test data

```{r, echo=FALSE}
table(Actual=test_data$target,Predicted=p_r_test)
```
**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(train_data$target,p_r_train)

```

**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(test_data$target,p_r_test)
```


####Conclusion for both logistic models: The PCA model is a simpler model as compared to the overall model and it deals well with multicollinear attributes but is not explicable and loses explainability.


**  **
**  **

###`Approach 4: Gradient boosting on Entire data`

```{r}
library(xgboost)
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

set.seed(134);x=sample(1:nrow(df),0.8*nrow(df))
tr=df[x,]
tst=df[-x,]

tr_label=tr$target
tr$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr),label=tr_label)

tst_label=tst$target
tst$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.005,"subsample"=0.8,"colsample_bytree"=0.6, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 1200)
pred_xgb_tr=predict(xgb_ml,xgb_tr); pred_xgb_tr=(ifelse(pred_xgb_tr>0.3,1,0))
pred_xgb_tst=predict(xgb_ml,xgb_tst);pred_xgb_tst=(ifelse(pred_xgb_tst>0.3,1,0))
```


> Confusion matrix on train data

```{r, echo=FALSE}
table(pred=pred_xgb_tr,Actual=tr_label)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(pred=pred_xgb_tst,Actual=tst_label)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(tr_label,pred_xgb_tr)
```


**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(tst_label,pred_xgb_tst)
```

**  **
**  **

###`Approach 5: Logistic regression with weights on PCA components built on numeric data`

```{r}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

numr_attr=sapply(df,is.integer)
numeric_data=df[,!numr_attr]
factor_data=as.data.frame(lapply(df[,numr_attr],as.factor))


prin_comp=prcomp(numeric_data,scale. = T)

pr_data=as.data.frame(prin_comp$x[,1:12])

pr_data$target=as.factor(df$target)

set.seed(134);x=sample(1:25000,0.8*25000)
train_data_num=pr_data[x,]
test_data_num=pr_data[-x,]

w=as.numeric(as.character(df$target[x]+1))


log_model=glm(target~.,data=train_data_num,family=binomial,weights =w)
#summary(log_model)
library(ROCR)
p <- predict(log_model,train_data_num, type="response")
pr <- prediction(p, train_data_num$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
p_r_train=(ifelse(p>0.3,1,0))
p <- predict(log_model,test_data_num, type="response")
p_r_test=(ifelse(p>0.3,1,0))
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf)
abline(a=0, b= 1)

```
> Confusion matrix on train data

```{r, echo=FALSE}
table(pred=p_r_train,Actual=train_data_num$target)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(pred=p_r_test,Actual=test_data_num$target)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(train_data_num$target,p_r_train)
```
**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(test_data_num$target,p_r_test)
```

**  **
**  **

###`Approach 6: Bagged Logistic regression on PCA components built on numeric data`

```{r, echo=TRUE, message=FALSE, warning=FALSE}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

numr_attr=sapply(df,is.integer)
numeric_data=df[,!numr_attr]
factor_data=as.data.frame(lapply(df[,numr_attr],as.factor))

prin_comp=prcomp(numeric_data,scale. = T)

pr_data=as.data.frame(prin_comp$x[,1:12])
pr_data$target=as.factor(df$target)

set.seed(134);x=sample(1:25000,0.8*25000)
train_data_num=pr_data[x,]
test_data_num=pr_data[-x,]
p_train=0
p_test=0
n=1000
for(i in 1:n)
{
  #print(i)
  set.seed(i)
  x=sample(1:nrow(train_data_num),0.5*nrow(train_data_num))
  log_model=glm(target~.,data=train_data_num[x,],family=binomial)
  p_train=p_train+predict(log_model,train_data_num[,-13], type="response")
  p_test=p_test+predict(log_model,test_data_num[,-13],type="response")
}
p_train=p_train/n
p_test=p_test/n
p_train=(ifelse(p_train>0.3,1,0))
p_test=ifelse(p_test>0.3,1,0)
```

> Confusion matrix on train data

```{r, echo=FALSE}
table(Actual=train_data_num$target,Predicted=p_train)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(Actual=test_data_num$target,Predicted=p_test)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(train_data_num$target,p_train)
```
**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(test_data_num$target,p_test)
```



**  **
**  **

###`Approach 7: Random forest on integer data with discrete values`

```{r, echo=TRUE, message=FALSE, warning=FALSE}
library(randomForest)
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

numr_attr=sapply(df,is.integer)
numeric_data=df[,!numr_attr]
factor_data=df[,numr_attr]
factor_data$target=as.factor(factor_data$target)

set.seed(134);x=sample(1:25000,0.8*25000)
train_data=factor_data[x,]
test_data=factor_data[-x,]
rf_model=randomForest(target~.,data=train_data)

pred_rf_train=predict(rf_model,train_data)
pred_rf_test=predict(rf_model,test_data)
```

> Confusion matrix on train data

```{r, echo=FALSE}
table(Predicted=pred_train,Actual=train_data$target)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(Actual=test_data$target,Predicted=pred_test)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(train_data$target,pred_train)
```
**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(test_data$target,pred_test)
```

**  **
**  **

###`Approach 8: Stacked xgboost with PCA comps + XGB,RF and LR predictions`

```{r, echo=TRUE}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

numr_attr=sapply(df,is.integer)
numeric_data=df[,!numr_attr]
factor_data=df[,numr_attr]
target=as.factor(factor_data$target)
set.seed(134);x=sample(1:25000,0.8*25000)
pca_ml=prcomp(numeric_data[x,],scale. = T)
pca_tr_data=as.data.frame(pca_ml$x[,1:12])
pca_tst_data=as.data.frame(predict(pca_ml,numeric_data[-x,]))[,1:12]
tr=data.frame(pca_tr_data,"rf"=as.numeric(pred_rf_train),"lg"=p_r_train,"xg"=pred_xgb_tr,"target"=target[x])
tst=data.frame(pca_tst_data,"rf"=as.numeric(pred_rf_test),"lg"=p_r_test,"xg"=pred_xgb_tst,"target"=target[-x])
log_model=glm(target~.,data=tr,family=binomial)
p_train <- predict(log_model,tr[,-ncol(tr)], type="response")
pred_stk_tr=(ifelse(p_train>0.3,1,0))

p_test <- predict(log_model,tst[,-ncol(tr)], type="response")
pred_stk_tst=(ifelse(p_test>0.3,1,0))

pr <- prediction(p_train, train_data_num$target)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf)
abline(a=0, b= 1)

```


> Confusion matrix on train data

```{r, echo=FALSE}
table(pred=pred_stk_tr,Actual=tr_label)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(pred=pred_stk_tst,Actual=tst_label)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(tr_label,pred_stk_tr)
```
**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(tst_label,pred_stk_tst)
```

**  **
**  **

###`Approach 9: Gradient boosting on 20 important xgboost features + 25 hidden from h2o`

```{r, message=FALSE, warning=FALSE}
library(xgboost)
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")

set.seed(134);x=sample(1:nrow(df),0.8*nrow(df))
tr=df[x,]
tst=df[-x,]

tr_label=tr$target
tr$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr),label=tr_label)

tst_label=tst$target
tst$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.005,"subsample"=0.8,"colsample_bytree"=0.6, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 1200)

pred_xgb_tr=predict(xgb_ml,xgb_tr); pred_xgb_tr=(ifelse(pred_xgb_tr>0.3,1,0))
pred_xgb_tst=predict(xgb_ml,xgb_tst);pred_xgb_tst=(ifelse(pred_xgb_tst>0.3,1,0))

xgb_imp=xgb.importance(feature_names = colnames(tr),model = xgb_ml)
top_20=as.character(unlist(xgb_imp[1:20,1]))
dt_train=data.frame(tr[,top_20])
dt_test=data.frame(tst[,top_20])

library(h2o)
h2o.init(nthreads = 3)
#h2o.shutdown()
dt_train.hex=as.h2o(df[x,],"train.hex")
dt_test.hex=as.h2o(df[-x,],"test.hex")
x1=names(dt_train)
y="target"

h2o_ml=h2o.deeplearning(x=x1,autoencoder = T,training_frame =  dt_train.hex ,activation = "Tanh",hidden = 25,epochs = 2000, seed = 123)
tr_features_h2o=as.data.frame(h2o.deepfeatures(h2o_ml, dt_train.hex))
tst_features_h2o=as.data.frame(h2o.deepfeatures(h2o_ml, dt_test.hex))

tr=data.frame(dt_train,tr_features_h2o,"target"=df$target[x])
tst=data.frame(dt_test,tst_features_h2o,"target"=df$target[-x])

tr_label=tr$target
tr$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr),label=tr_label)

tst_label=tst$target
tst$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.005,"subsample"=0.8,"colsample_bytree"=0.8, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 1200)

pred_xgb_tr=predict(xgb_ml,xgb_tr); pred_xgb_tr=(ifelse(pred_xgb_tr>0.3,1,0))
pred_xgb_tst=predict(xgb_ml,xgb_tst);pred_xgb_tst=(ifelse(pred_xgb_tst>0.3,1,0))

```



> Confusion matrix on train data

```{r, echo=FALSE}
table(pred=pred_xgb_tr,Actual=tr_label)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(pred=pred_xgb_tst,Actual=tst_label)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(tr_label,pred_xgb_tr)
```


**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(tst_label,pred_xgb_tst)
```



**  **
**  **

###`Approach 10: K-means + xgboost`

```{r, echo=TRUE, message=FALSE, warning=FALSE}
df <- read.csv("C:/Users/mukes_000/Downloads/Telecom_Churn_Data/Telecom Churn Data/Data.csv")
target=df$target
df$target=NULL

df=scale(df)
set.seed(123)
kmeans_ml=kmeans(df,3,iter.max = 100)
df=data.frame(df,target)

library(xgboost)
df$cluster=kmeans_ml$cluster
df_1=df[df$cluster==1,]
set.seed(134);x=sample(1:nrow(df_1),0.8*nrow(df_1))
tr_1=df_1[x,]
tst_1=df_1[-x,]

tr_1$cluster=NULL
tst_1$cluster=NULL

tr_label=tr_1$target
tr_1$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr_1),label=tr_label)

tst_label=tst_1$target
tst_1$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst_1),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.005,"subsample"=0.8,"colsample_bytree"=0.6, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 2000)

pred_xgb_tr_1=predict(xgb_ml,xgb_tr); pred_xgb_tr_1=(ifelse(pred_xgb_tr_1>0.3,1,0))
pred_xgb_tst_1=predict(xgb_ml,xgb_tst);pred_xgb_tst_1=(ifelse(pred_xgb_tst_1>0.3,1,0))
tr=data.frame("actual"=tr_label,"predicted"=pred_xgb_tr_1)
tst=data.frame("actual"=tst_label,"predicted"=pred_xgb_tst_1)
#####CLUSTER-2
df_1=df[df$cluster==2,]
set.seed(134);x=sample(1:nrow(df_1),0.8*nrow(df_1))
tr_1=df_1[x,]
tst_1=df_1[-x,]

tr_1$cluster=NULL
tst_1$cluster=NULL

tr_label=tr_1$target
tr_1$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr_1),label=tr_label)

tst_label=tst_1$target
tst_1$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst_1),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.005,"subsample"=0.8,"colsample_bytree"=0.6, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 2000)

pred_xgb_tr_2=predict(xgb_ml,xgb_tr); pred_xgb_tr_2=(ifelse(pred_xgb_tr_2>0.3,1,0))
pred_xgb_tst_2=predict(xgb_ml,xgb_tst);pred_xgb_tst_2=(ifelse(pred_xgb_tst_2>0.3,1,0))
tr_2=data.frame("actual"=tr_label,"predicted"=pred_xgb_tr_2)
tst_2=data.frame("actual"=tst_label,"predicted"=pred_xgb_tst_2)
tr=rbind(tr,tr_2)
tst=rbind(tst,tst_2)
#######Cluster - 3
df_1=df[df$cluster==3,]
set.seed(134);x=sample(1:nrow(df_1),0.8*nrow(df_1))
tr_1=df_1[x,]
tst_1=df_1[-x,]

tr_1$cluster=NULL
tst_1$cluster=NULL

tr_label=tr_1$target
tr_1$target=NULL
xgb_tr=xgb.DMatrix(as.matrix(tr_1),label=tr_label)

tst_label=tst_1$target
tst_1$target=NULL
xgb_tst=xgb.DMatrix(as.matrix(tst_1),label=tst_label)

xgb_ml=xgb.train(params=list("eta"=0.5,"subsample"=0.8,"colsample_bytree"=0.6, "max_depth"=5,
                             "objective"="binary:logistic"),data=xgb_tr,nrounds = 500)

pred_xgb_tr_3=predict(xgb_ml,xgb_tr); pred_xgb_tr_3=(ifelse(pred_xgb_tr_3>0.3,1,0))
pred_xgb_tst_3=predict(xgb_ml,xgb_tst);pred_xgb_tst_3=(ifelse(pred_xgb_tst_3>0.3,1,0))
tr_3=data.frame("actual"=tr_label,"predicted"=pred_xgb_tr_3)
tst_3=data.frame("actual"=tst_label,"predicted"=pred_xgb_tst_3)
tr=rbind(tr,tr_3)
tst=rbind(tst,tst_3)

```



> Confusion matrix on train data

```{r, echo=FALSE}
table(pred=tr$predicted,Actual=tr$actual)
```

> Confusion matrix on test data

```{r, echo=FALSE}
table(pred=tst$predicted,Actual=tst$actual)
```


**  **

**`Metrics on train data`** 

```{r, echo=FALSE}
metrics(actual = tr$actual,predicted = tr$predicted)
```


**  **

**`Metrics on test data`** 

```{r, echo=FALSE}
metrics(tst$actual,tst$predicted)
```


