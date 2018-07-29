#################### STEP 2 ###############################################################################


#is there anything to be gained in adding a field with the titles of passagers ?

#Some data wrangling/munging is required, specifically text processing.

#extract title
titles <- testtrain1$Name %>% 
  as.character %>% 
  sapply(function(x) strsplit(x,", ")[[1]][2]) %>%
  sapply(function(x) strsplit(x,". ")[[1]][1]) %>%
  unname

#what are the most common titles ?
titles
titles %>% table
titles %>% table %>% sort

# military, medical and religious titles 
MilDocRev <- as.integer(titles %in% c("Capt","Col","Dr","Major","Rev"))
Military <- as.integer(titles %in% c("Capt","Col","Major"))
Doctor <- as.integer(titles == "Dr")
Reverend <- as.integer(titles == "Rev")
Noble <- as.integer(titles %in% c("Dona","Lady","th","Don","Jonkheer","Sir"))

# Map all foreign, noble, military, religious, medical titles to Mr, Mrs, Miss or Master
titles[titles %in% c("Dona","Lady","Mme","th")] <- "Mrs"
titles[titles %in% c("Mlle","Ms")] <- "Miss"
titles[titles %in% c("Don","Capt","Col","Dr","Jonkheer","Major","Rev","Sir")] <- "Mr"


testtrain2 <- cbind(testtrain1,titles,MilDocRev,Military,Doctor,Reverend,Noble) #expand the current "testtrain" repository of all data, adding the new title fields.
train2 <- testtrain2[testtrain2$Survived!=-2000,] # extract training data from the combined data. Use logical indexing to take all record with a 1 or 0 in the "Survived field (test data has -2000 in that field)
names(train2) #examine the field names of train2
trainnames2 <- c(trainnames,"titles","MilDocRev","Military","Doctor","Reverend","Noble")
X2 <- train2[,trainnames2] # grab the fields used in the previous example, as well as the new title fields.

#run random forest as before, on new data
RF2 <- randomForest(X2,as.factor(Y),do.trace=T,importance=T,ntree=200)
auc(Y,RF2$votes[,2])
varImpPlot(RF2)

xyhat <- cbind(X2,yhat=RF2$votes[,2])  #combine data and predictions. This is useful for visualisation

convelem <- function(X) 
  model.matrix( ~ .,data=X)[,-1]

X2dummy <- convelem(X2) #create dummy vector using convelem function 

# run linear model as before
cvg2 <- cv.glmnet(as.matrix(X2dummy),as.factor(Y),family="binomial",type.measure="auc",nfolds = 10)
plot(cvg2)
max(cvg2$cvm)  #best AUC for linear model

cvg2 <- cv.glmnet(as.matrix(X2dummy),as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=0.5)
plot(cvg2)
max(cvg2$cvm)  #best AUC for linear model

#model parameters for bet AUC, and model of 1se difference 
cvg2$glmnet.fit$beta[,which(cvg2$lambda==cvg2$lambda.min)]
cvg2$glmnet.fit$beta[,which(cvg2$lambda==cvg2$lambda.1se)]

###  AUC improves, titles do matter

# density plot of yhat, facet by pclass and title
# 
#ggraptR()


#grid search

nsample = 40
alpha <- seq(0,1,length=11)
maxauc <- matrix(0,ncol=length(alpha),nrow=nsample)
colnames(maxauc) <- alpha
for(j in 1:nsample) {
  for(i in  seq_along(alpha))
  {  
    cvg1 <- cv.glmnet(X2dummy,as.factor(Y),family="binomial",type.measure="auc",nfolds = 10,alpha=alpha[i])
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

save.image("step2save.RData")


####################### STEP 2 ##########################################
