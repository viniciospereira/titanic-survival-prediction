
################### STEP 4 : functionalisation of model building  ##############################
##########  CREATE SOME FUNCTIONS TO MAKE LIFE EASIER - WE ARE REPEATING THE SAME CODE A LOT !

buildRF <- function(XX,YY,ntree=200,verb=50)
{
  RF <- randomForest(XX,as.factor(YY),do.trace=verb,importance=T,ntree=ntree)
  rfauc <- auc(YY,RF$votes[,2])
  varImpPlot(RF)
  return(list(RF=RF,rfauc=rfauc))
}  

RFb <- buildRF(X3,Y)
RFb$rfauc


buildGLMNET <- function(XXdummy,YY,nfolds=10)
{
  cvg <- cv.glmnet(as.matrix(XXdummy),as.factor(YY),family="binomial",type.measure="auc",nfolds = nfolds,keep=T)
  plot(cvg)
  minbetas=cvg$glmnet.fit$beta[,which(cvg$glmnet.fit$lambda==cvg$lambda.min)]
  se1betas=cvg$glmnet.fit$beta[,which(cvg$glmnet.fit$lambda==cvg$lambda.1se)]
  return(list(cvg=cvg,minbetas=minbetas,se1betas=se1betas))
}  

GLMb <- buildGLMNET(X3dummy,Y)
GLMb$minbetas
GLMb$se1betas


############ interaction features for glmnet
makeinteractions <- function(X)
{
  for(i in 1:ncol(X))
    for(j in 1:i)
    {
      X <- cbind(X,X[,i]*X[,j])
      if(i == j)
        iname = paste(colnames(X)[i],"_squared",sep="")
      else
        iname = paste(colnames(X)[i],"_",colnames(X)[j],sep="")
      colnames(X)[ncol(X)] <- iname
    }  
  return(X)
}  


X1inter <- makeinteractions(X1dummy)
X3inter <- makeinteractions(X3dummy)


buildGLMNETinteractions <- function(XXdummy,YY,nfolds=10)
{
  
  XXdummyint <- makeinteractions(XXdummy)
  cvg <- cv.glmnet(as.matrix(XXdummyint),as.factor(YY),family="binomial",type.measure="auc",nfolds = 10,keep=T)
  plot(cvg)
  minbetas=cvg$glmnet.fit$beta[,cvg$glmnet.fit$lambda==cvg$lambda.min]
  se1betas=cvg$glmnet.fit$beta[,cvg$glmnet.fit$lambda==cvg$lambda.1se]
  return(list(cvg=cvg,minbetas=minbetas,se1betas=se1betas))
}  

bglmi <- buildGLMNETinteractions(X1dummy,Y,nfolds=10)

bglmi <- buildGLMNETinteractions(X3dummy,Y,nfolds=10)


buildGLMNET2 <- function(XX,YY,nfolds=10,interactions=F)
{
  XXdummy <- convelem(XX)
  if(interactions) XXdummy <- makeinteractions(XXdummy) 
  cvg <- cv.glmnet(XXdummy,as.factor(YY),family="binomial",type.measure="auc",nfolds = nfolds,keep=T)
  plot(cvg)
  minbetas=cvg$glmnet.fit$beta[,which(cvg$glmnet.fit$lambda==cvg$lambda.min)]
  se1betas=cvg$glmnet.fit$beta[,which(cvg$glmnet.fit$lambda==cvg$lambda.1se)]
  return(list(cvg=cvg,minbetas=minbetas,se1betas=se1betas))
}  

GLMb2 <- buildGLMNET2(X3,Y)
GLMb2$minbetas
GLMb2$se1betas

GLMb2int <- buildGLMNET2(X3,Y,interactions=T)
GLMb2int$minbetas
GLMb2int$se1betas



xgprep <- function(X,Y=NULL)
  if(!is.null(Y))
    return(xgb.DMatrix(Matrix(as.matrix(X),sparse=T), label = Y)) else
      return(xgb.DMatrix(Matrix(as.matrix(X),sparse=T)))


buildXGBOOST <- function(XXdummy,YY,colsample=0.5,subsample=0.7,max.depth =4, eta = 0.01,alpha=0,nfold = 10,verb=T,nround=2000,importance=F)
{
  
  traindata <- xgprep(XXdummy,YY)
  param <- list("objective" = "binary:logistic", #"reg:linear",
                "eval_metric" = "auc",
                colsample=colsample,subsample=subsample,max.depth =max.depth, eta = eta,alpha=alpha)
  history <- xgb.cv(data = traindata, nround=nround, nthread = 2, nfold = nfold,
                    params=param,prediction=TRUE,verbose = verb)
  xgbauc <- max(history$evaluation_log$test_auc_mean)  
  itermax <- history$evaluation_log$test_auc_mean %>% which.max
  itermax
  plot(history$evaluation_log$test_auc_mean)
  history <- xgb.cv(data = traindata, nround=itermax, nthread = 2, nfold = nfold,
                    params=param,prediction=TRUE,verbose = T)
  xgbauc2 <- auc(YY,history$pred) 
  xgbauc2
  
  if(importance)
  { 
    xgbmod <- xgb.train(data = traindata, nround=itermax, nthread = 2,
                        params=param,prediction=TRUE,verbose = T)
    importances <- xgb.importance(colnames(XXdummy),xgbmod)
    print(importances)
    
  }  else { importances = NULL; xgbmod=NULL }
  print(class(xgbmod))
  return(list(history=history,xgbauc=xgbauc,xgbauc2=xgbauc2,itermax=itermax,importances=importances,xgbmod=xgbmod))
  
}  

bxgb <- buildXGBOOST(X3dummy,Y)

bxgbImp <- buildXGBOOST(X3dummy,Y,importance = T,nround=1000)
bxgbImp$importances
bxgbImp$importances %>% arrange(-Cover)

xgb.model.dt.tree(colnames(X3dummy),bxgbImp$xgbmod)

runmodels <- function(XX,YY, ntree=200,type.measure='auc',verb=T,nfolds=10,runinteractions=T,showinteractions=F)
{
  #create dummy variables. We will need these for glmnet and xgboost
  XXdummy <- convelem(XX)
  
  # run random forest, plot importances, return model and AUC
  RFlist <- buildRF(XX,YY,ntree=ntree,verb=verb)
  RF <- RFlist$RF
  rfauc <- RFlist$rfauc
  xyhat <- cbind(XX,yhat=RF$predicted)
  
  # run glmnet with simple features, plot AUC vs lambda, return model and AUC
  GLMlist <- buildGLMNET(XXdummy,YY,nfolds=nfolds)
  cvg <-  GLMlist$cvg
  minbetas <- GLMlist$minbetas
  se1betas <- GLMlist$se1betas
  
  if(runinteractions)
  {  
    GLMinterlist <- buildGLMNETinteractions(XXdummy,YY,nfolds=nfolds)
    cvginteract <-  GLMinterlist$cvg
    minbetasinteract <- GLMinterlist$minbetas
    se1betasinteract <- GLMinterlist$se1betas
  }
  cvxgb <- buildXGBOOST(XXdummy,YY,colsample=0.5,subsample=0.7,max.depth =4, eta = 0.01,alpha=0,nfold = 10,verb=verb)
  history <- cvxgb$history
  xgbauc <- cvxgb$xgbauc
  xgbauc2 <- cvxgb$xgbauc2
  
  #traindata <- xgprep(XXdummy,YY)
  #param <- list("objective" = "binary:logistic", #"reg:linear",
  #              "eval_metric" = "auc",
  #              colsample=0.5,subsample=0.7,max.depth =4, eta = 0.01,alpha=0)
  #history <- xgb.cv(data = traindata, nround=1000, nthread = 2, nfold = 10,
  #                  params=param,prediction=TRUE,verbose = verb)
  #xgbauc <- max(history$dt$test.auc.mean)
  
  
  
  if(verb)
  {  
    print(c("RF auc: ",rfauc))
    print(c("cvglmnet auc",max(cvg$cvm)))
    
    print(c("xgbauc auc",xgbauc))
    print(c("xgbauc2 auc",xgbauc2)) 
    print(names(XXdummy))
    
  }
  
  if(runinteractions)
  {  
    if(verb)
    {  
      print(minbetas)
      print(se1betas)
      print(c("cvglmnet_interaction auc",max(cvginteract$cvm))) 
    }  
    if(showinteractions)
    {  
      print(minbetasinteract)
      print(se1betasinteract)
    }
    return(list(RF=RF,rfauc=rfauc,xyhat=xyhat,
                XXdummy=XXdummy,
                cvglm=cvg,minbetas=minbetas,se1betas=se1betas,
                history=history,xgbauc=xgbauc,xgbauc2=xgbauc2,
                cvglminteract=cvginteract,minbetasinteract=minbetasinteract,se1betasinteract=se1betasinteract))  
  }  
  else
  {  
    if(verb)
    {  
      print(minbetas)
      print(se1betas)
    }  
    return(list(RF=RF,rfauc=rfauc,xyhat=xyhat,
                XXdummy=XXdummy,
                cvglm=cvg,minbetas=minbetas,se1betas=se1betas,
                history=history,xgbauc=xgbauc,xgbauc2=xgbauc2))
  }             
}  


#run all models with interactions on most recently processed data set with one call
Run3 <- runmodels(X3,Y,verb=T)

Run3i <- runmodels(X3,Y,verb=T,runinteractions=T)


################### END STEP 4 ############################### 

save.image("step4save.RData")

