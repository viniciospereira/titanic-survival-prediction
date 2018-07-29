
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


train <- read.csv("train.csv")
test <- read.csv("test.csv")

testY <- test %>% 
         mutate(Survived=-2000) %>% 
         select(PassengerId,Survived,Pclass:Embarked)  #add an outcome code to the test set

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
Y <- testtrain1 %>% filter(Survived != -2000) %>% select(Survived) %>% .[[1]]

X1Y <- cbind(X1,Y)

# build a Random Forest model of the data
RF1 <- randomForest(as.data.frame(X1),as.factor(Y),do.trace=50,importance=T,ntree=200)

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
  
  if(auc) { print("auc"); auc(Y,Yhat_ranger) %>% print }
  if(mse) { print("mse"); mse(Y,Yhat_ranger) %>% print }
  if(mae) {print("mae"); mae(Y,Yhat_ranger) %>% print }
  if(rsquared) {print("rsquared"); rsquared(Y,Yhat_ranger) %>% print }
  if(misclass_rate) {print("misclass_rate"); misclass_rate(Y,Yhat_ranger) %>% print }
  if(accuracy) {print("accuracy"); accuracy(Y,Yhat_ranger) %>% print }
  if(logloss) {print("logloss"); logloss(Y,Yhat_ranger) %>% print }
}

resreport(Y,Yhat)
resreport(Y,Yhat_ranger)

#error meaure with a string for the function name
err_measure <- function(err_string,Y,Yhat) {
  exstr <- str_c(err_string,"(Y,Yhat)")
  outval <- eval(parse(text=exstr))  #this is how to execute a text string in R
  names(outval) <- err_string
  outval
}

err_measure("auc",Y,Yhat_ranger)

#let's test it

errlist <- list("auc","mse","mae","rsquared","misclass_rate","accuracy","logloss")
errlist %>% resreport(Y,Yhat_ranger)

#new error functions
maxerror <- function(Y,Yhat) max(abs(Y-Yhat))
corerror <- function(Y,Yhat) cor(Y,Yhat)
errlist2 <- c(errlist,"maxerror","corerror")

#run the report with new error functions
errlist2 %>% resreport2(Y,Yhat_ranger)



#how fast were they ?
# build a Random Forest model of the data
system.time(RF1 <- randomForest(as.data.frame(X1),as.factor(Y),do.trace=50,importance=T))

#build a ranger random forest
system.time(ran1 <- ranger(Y ~ ., data=X1Y,importance="permutation"))

varImpPlot(RF1)  # show variable importances in RF model
ran1$variable.importance %>% sort %>% barplot(horiz=T,cex.names=0.5) # show variable importance of ranger model : permutation

X1Yhat <- cbind(X1,Yhat_ranger)

#what if we just used the 4 most significant variables ?
X1reduced <- transmute(X1,Sex,Pclass,Fare,Age)

RF1reduced <- randomForest(X1reduced,as.factor(Y),do.trace=50,importance=T,ntree=200)
errlist2 %>% resreport2(Y,RF1reduced$votes[,2])  #report the AUC - Area Under (the ROC) Curve


varImpPlot(RF1reduced)  # show variable importances in RF model

# now try a (generalised, regularised) linear model with glmnet
#need to convert categorics to elementary for linear model


X1dummy <- model.matrix( ~ ., X1)[,-1]

#build a binomial generalised linear model using lasso and ridge regression regularisation
#and n-fold cross-validation
cvg1 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,keep=T)
#what if we vary alpha ?
cvg0.5 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=0.5,keep=T)

#indices of the best glmnet model
optind <- which(cvg1$lambda == cvg1$lambda.min)
se1ind <- which(cvg1$lambda == cvg1$lambda.1se)
#best predictions

#plot the model
plot(cvg1)

# what is the AUC of the best model ?
max(cvg1$cvm)



# what are the parameters of the best model, and the 1se model ?
cvg1$glmnet.fit$beta[,which(cvg1$lambda==cvg1$lambda.min)]
cvg1$glmnet.fit$beta[,which(cvg1$lambda==cvg1$lambda.1se)]

fullreport <- function(error_list,Ylist,Yhatlist,NameList) {
  resmat <- list(Ylist,Yhatlist) %>% pmap(resreport,funclist=error_list) %>% reduce(rbind) 
  row.names(resmat) <- NameList
  resmat
}
Ylist <- list(Y,Y,Y,Y,Y)
Yhat_list <- list(Yhat,Yhat_ranger,RF1reduced$votes[,2],cvg1$fit.preval[,optind],cvg1$fit.preval[,se1ind])
NameList <- str_c("X1 - median imputation: ",c("randomForest","ranger","randomForest - top4 fields","Lasso AUC-optimal Model","Lasso AUC-opitmal 1SE simplified")) %>% 
           as.list

errlist2 %>% fullreport(Ylist,Yhat_list,NameList)  


#grid search

nsample = 40
alpha <- seq(0,1,length=11)
maxauc <- matrix(0,ncol=length(alpha),nrow=nsample)
colnames(maxauc) <- alpha
for(j in 1:nsample) {
for(i in  seq_along(alpha))
  {  
    cvg1 <- cv.glmnet(X1dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=alpha[i])
    maxauc[j,i] <- max(cvg1$cvm)
  }
  maxauc_gathered <- gather(maxauc %>% as.data.frame,key = alpha,value= auc) %>% filter(auc!=0)
  z <- ggplot(maxauc_gathered, aes(y=auc, x=as.factor(alpha))) + 
    geom_boxplot(aes(fill=as.factor(alpha)), stat="boxplot", position="dodge", alpha=0.5, width=0.2) + 
    geom_violin(aes(fill=as.factor(alpha)), stat="ydensity", position="dodge", alpha=0.5, trim=TRUE, scale="area") + 
    #coord_flip() + theme_grey() + 
    theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + 
    guides(fill=guide_legend(title="alpha")) + xlab("as.factor(alpha)") + ylab("auc") 
  print(z)
  
  print(j)
}  



###########################  END OF STEP 1 ####################################################

save.image("step1save.RData")
