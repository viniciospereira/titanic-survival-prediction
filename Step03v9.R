

######################## STEP 3 : Ethicity ###############################
####### Let's add ethnicity into the picture: are there racial differences in outcomes ?


testtrain3A <- read.csv("titanicWithEthnicity.csv")  #a new dataset, with an ethnicity field
testtrain3 <- cbind(testtrain3A,titles,MilDocRev,agemissing=testtrain2$agemissing,select(testtrain2,Military,Doctor,Reverend,Noble)) #add the new titles fields to the new data set
train3 <- testtrain3[testtrain3$Survived !=-2000,-1] #take only the training data, remove the 1st column of row labels
trainnames3 <- c(trainnames2,"Ethnicity")
X3a <- train3[,trainnames3] #take only the fields we want to use for modelling

ageimputeMedian <- function(x) {
  
  xx <- x 
  xx[] <- x %>% lapply(function(x) if(is.factor(x)) x else {x[is.na(x)] <- median(x,na.rm=T);x})
  xx
}  
X3 <- X3a %>% ageimputeMedian #call the median impute function to impute age as before

# build some models as before
RF3 <- randomForest(X3,as.factor(Y),do.trace=50,importance=T,ntree=200) #random forest
auc(Y,RF3$votes[,2])  #AUC of Random Forest
varImpPlot(RF3) #RF variable importances

xyhat <- cbind(X3,yhat=RF3$votes[,2])  #data for examination with ggraptR
X3dummy <- convelem(X3) #dummy fields for glmnet model

cvg3 <- cv.glmnet(as.matrix(X3dummy),as.factor(Y),family="binomial",type.measure="auc",nfolds = 10)
plot(cvg3) #plot cross-validation results
max(cvg3$cvm) #return AUC of best linear model

cvg3$glmnet.fit$beta[,which(cvg3$glmnet.fit$lambda==cvg3$lambda.min)] #params of model with highest AUC
cvg3$glmnet.fit$beta[,which(cvg3$glmnet.fit$lambda==cvg3$lambda.1se)] #params of simpler model model with AUC within 1 standard error of best model

#run xgboost model
traindata <- xgb.DMatrix(Matrix(as.matrix(X3dummy),sparse=T), label = Y)
param <- list("objective" = "binary:logistic", 
              "eval_metric" = "auc",
              colsample=0.5,subsample=0.7,max.depth =4, eta = 0.01,alpha=0)
history <- xgb.cv(data = traindata, nround=3000, nthread = 2, nfold = 10,
                  params=param,prediction=TRUE,verbose = T)

plot(history$evaluation_log$test_auc_mean) 

xgbauc <- max(history$evaluation_log$test_auc_mean) 
xgbauc
itermax <- history$evaluation_log$test_auc_mean %>% which.max
itermax
history <- xgb.cv(data = traindata, nround=itermax, nthread = 2, nfold = 10,
                  params=param,prediction=TRUE,verbose = T)
xgbauc2 <- auc(Y,history$pred) 
xgbauc2

history$evaluation_log$test_auc_mean[itermax]


save.image("step3save.RData")


##### END STEP 3 ##########################
