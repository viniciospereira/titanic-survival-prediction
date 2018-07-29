
################## STEP 1 : intialisation, data ingestion and first model ######################

rm(list=ls())

install.packages("pacman")
library(pacman)

p_load(magrittr)
p_load(ggraptR)
p_load(randomForest)
p_load(glmnet)
p_load(ggplot2)
p_load(xgboost)
p_load(GGally)
p_load(tidyverse)
p_load(ranger)
p_load(rattle)


train <- read.csv("train.csv")
test <- read.csv("test.csv")

testY <- test %>% 
         mutate(Survived=-2000) %>% 
         select(PassengerId,Survived,Pclass:Embarked)  #add an outcome code to the test set


#select(mutate(test,Survived=-2000),PassengerId,Survived,Pclass:Embarked)

names(testY)==names(train) #are the names the same ?

#Don't forget to combine test and training
### check for new categories, missing values
### ideally pre-process, impute them together
testtrain <- rbind(train,testY) #testY and train have the same columns. Combine them into a single data set.
# this will be useful later.
# save the combined data set.
#write.csv(testtrain,"titanic.csv")

#Explore the data
summary(testtrain)
#ggraptR()


#what if we ignore the test set, and only use the training set ?

#build first model - only fields that don't need data "wrangling"/"munging"
Y <- train$Survived
trainnames <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")
X1 <- testtrain %>% filter(Survived != -2000) %>% select(trainnames)   #select_(.dots=trainnames)

X1a <- X1 %>% as.data.frame
#have a look at pair plots:
#ggpairs(X1a)
#ggpairs(cbind(X1a,Y=as.factor(Y)),mapping=ggplot2::aes(colour=Y,alpha=0.5))

# do we have missing values ?
summary(X1) 
# yes we do - Age !

#simple imputation of age - using only training data
agemissing <- is.na(X1$Age)
X1keepmissing <- X1
X1$Age[agemissing] <- median(X1$Age,na.rm=T) #fill missing age entries with median of non-missing entries
X1 <- cbind(X1,agemissing)   #add a missing indicator field
trainnames <- c(trainnames,"agemissing")


#sligthly better imputation of age: use the entire data set !
#we can do this without fear of contamination / overfitting : we are not using the Y values.

#also impute fare

testtrain1 <- testtrain %>% 
              mutate(agemissing=is.na(Age)  %>%  
                      as.numeric, Age=ifelse(agemissing,median(.$Age,na.rm=T),Age)) %>% 
              mutate(Fare=ifelse(is.na(Fare),median(.$Fare,na.rm=T),Fare)) 
              

#take imputed training set

X1 <- testtrain1 %>% filter(Survived != -2000) %>% select(trainnames)
Y <- testtrain1 %>% filter(Survived != -2000) %>% pull(Survived)   #select(Survived) %>% .[[1]]

X1Y <- cbind(X1,Y)

# build a Random Forest model of the data
RF1 <- randomForest(X1,as.factor(Y),do.trace=50,importance=T,ntree=200)

#build a ranger random forest
ran1 <- ranger(Y ~ ., data=X1Y,importance="permutation")

# error measures

#error functions
mae <- function(Y,Yhat) (Y - Yhat) %>% abs %>% mean
mse <- function(Y,Yhat) (Y - Yhat)^2 %>% mean
rsquared <- function(Y,Yhat) 1- (mse(Y,Yhat)/var(Y))
misclass_rate <- function(Y,Yhat) {  
  misclass_table <- ((Yhat >= 0.5 & Y) | (Yhat < 0.5 & !Y)) %>% table
  (misclass_table["FALSE"]/length(Y)) %>% unname
}
accuracy  <- function(Y,Yhat) 1-misclass_rate(Y,Yhat)
logloss <- function(Y,Yhat) -ifelse (Y==1,ifelse(Yhat!=0,log(Yhat),0),ifelse(Yhat!=1,log(1-Yhat),0)) %>% mean

Yhat <- RF1$votes[,2]
Yhat_ranger <- ran1$predictions

#randomForest results 
auc(Y,Yhat)
mae(Y,Yhat)
mse(Y,Yhat)
rsquared(Y,Yhat)
misclass_rate(Y,Yhat)
accuracy(Y,Yhat)
logloss(Y,Yhat)

#ranger results
auc(Y,Yhat_ranger)
mae(Y,Yhat_ranger)
mse(Y,Yhat_ranger)
rsquared(Y,Yhat_ranger)
misclass_rate(Y,Yhat_ranger)
accuracy(Y,Yhat_ranger)
logloss(Y,Yhat_ranger)

resreport <- function(Y,Yhat,auc=T,mae=T,mse=T,rsquared=T,misclass_rate=T,accuracy=T,logloss=T) {
  
  if(auc) { print("auc"); auc(Y,Yhat) %>% print }
  if(mse) { print("mse"); mse(Y,Yhat) %>% print }
  if(mae) {print("mae"); mae(Y,Yhat) %>% print }
  if(rsquared) {print("rsquared"); rsquared(Y,Yhat) %>% print }
  if(misclass_rate) {print("misclass_rate"); misclass_rate(Y,Yhat) %>% print }
  if(accuracy) {print("accuracy"); accuracy(Y,Yhat) %>% print }
  if(logloss) {print("logloss"); logloss(Y,Yhat) %>% print }
}

resreport(Y,Yhat)
resreport(Y,Yhat_ranger)

#how fast were they ?
# build a Random Forest model of the data
system.time(RF1 <- randomForest(as.data.frame(X1),as.factor(Y),do.trace=50,importance=T))

#build a ranger random forest
system.time(ran1 <- ranger(Y ~ ., data=X1Y))

varImpPlot(RF1)  # show variable importances in RF model
ran1$variable.importance %>% sort %>% barplot(horiz=T,cex.names=0.5) # show variable importance of ranger model : permutation

cor(X1$Pclass,X1$Fare)

X1Yhat <- cbind(X1,Yhat_ranger)

#what if we just used the 4 most significant variables ?
X1reduced <- transmute(X1,Sex,Pclass,Fare,Age)

RF1reduced <- randomForest(X1reduced,as.factor(Y),do.trace=50,importance=T,ntree=200)
resreport(Y,RF1reduced$votes[,2])  #report the AUC - Area Under (the ROC) Curve


varImpPlot(RF1reduced)  # show variable importances in RF model

# now try a (generalised, regularised) linear model with glmnet
#need to convert categorics to elementary for linear model


X1dummy <- model.matrix( ~ ., X1)[,-1]

#build a binomial generalised linear model using lasso and ridge regression regularisation
#and n-fold cross-validation
cvg1 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10)
#what if we vary alpha ?
cvg1 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=0.5)

#plot the model
plot(cvg1)

# what is the AUC of the best model ?
max(cvg1$cvm)



# what are the parameters of the best model, and the 1se model ?
cvg1$glmnet.fit$beta[,which(cvg1$lambda==cvg1$lambda.min)]
cvg1$glmnet.fit$beta[,which(cvg1$lambda==cvg1$lambda.1se)]

#grid search

nsample = 40
alpha <- seq(0,1,length=11)
maxauc <- matrix(0,ncol=length(alpha),nrow=nsample)

for(j in 1:nsample)
for(i in  seq_along(alpha))
{  
 cvg1 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=alpha[i])
 maxauc[j,i] <- max(cvg1$cvm)
 print(paste(j,i))
}

colnames(maxauc) <- paste0("alpha",alpha)

matplot(alpha,maxauc %>% t)

plot(alpha,apply(maxauc,2,mean))

plot(alpha,apply(maxauc,2,function(x) mean(x)-sd(x)))

###########################  END OF STEP 1 ####################################################

save.image("step1save.RData")
