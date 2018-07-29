
#################STEP 8 - a better way to impute missing values! ################
################### impute age PREDICTIVELY

# we can use the old train6 - no new columns will be added this time
# we will use the entire data set, test and training, becuase we can !
# unlabelled data can be useful in feature engineering. This can be done
# with much reduced fear of overfitting, 
# because we are NOT touching the target field during feature engineering

testtrain7 <- testtrain6
train7 <- train6

X7 <-  train6[,trainnames6]
X7keepmissing <- X7
agemissing7 <- is.na(X7$Age)
#note that there is a missing fare. We will need to deal with it.
# there is only one, so we will just get rid of it in the imputation model
summary(testtrain7)
#create an "X" set for predictive imputation : we will use all predictors except Age, where Age is not missing
ageind <- which(names(testtrain7)=="Age") #use this to find the Y value
agenameind <- which(trainnames6=="Age") #use this to exclude Age from imputation predictors
ageprednames <- trainnames6[-agenameind]
AgeFareNotMissing <- !is.na(testtrain7$Age) & !is.na(testtrain7$Fare)
Xage <- testtrain7[AgeFareNotMissing,ageprednames]
# create a "Y" target set for predictive imputation of age: all non-missing age entries, in test and training.
Yage <- testtrain7[AgeFareNotMissing,ageind]
RFage <- randomForest(Xage,Yage,do.trace=100,importance=T,ntree=2000)#build the imputation model
varImpPlot(RFage)


#titles field takes care of half of the error explained
RFageTitlesOnly <- randomForest(Xage["titles"],Yage,do.trace=100,importance=T,ntree=2000)#build the imputation model

#  Xage and X7[,-3]  have the same columns
AgeImpute <- predict(RFage,X7[agemissing7,-3])#impute age predictively using the RF imputation model
X7[agemissing7,3] <- AgeImpute #replace missing age values with imputed values.
Run7 <- runmodels(X7,Y,runinteractions = T)

save.image("step8save.RData")



############### END STEP 8 ################################

