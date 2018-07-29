+
######### STEP10  ######### one-to-many features  #########################################

#group by last name
lastname <- testtrain$Name %>% as.character() %>% unname %>% 
  sapply(function(x) strsplit(x,", ")[[1]][1]) %>% unname

testtrain8 <- cbind(testtrain7,lastname)

grouptt8 <- group_by(testtrain8,lastname,Pclass)

#with Embarked
#grouptt8 <- group_by(testtrain8,lastname,Pclass,Embarked)

# count titles per family
namesbytitle <- grouptt8 %>% 
  do(data.frame(TitleCounts=table(.$titles))) %>% 
  spread(TitleCounts.Var1,TitleCounts.Freq) 

#break up counts by total, males, females, adults and children    
familytotal <- namesbytitle[,3:6]  %>% as.data.frame %>% apply(1,sum)
familymales <-  namesbytitle[,c(3,5)] %>% as.data.frame %>% apply(1,sum)
familyfemales <- namesbytitle[,c(4,6)] %>% as.data.frame %>% apply(1,sum)
familychildren <- namesbytitle[,3:4] %>% as.data.frame %>% apply(1,sum)
familyadults <- namesbytitle[,5:6] %>% as.data.frame %>% apply(1,sum)
family_kids_to_adults <- familychildren/familyadults
#family_females_to_males <- familymales/familyfemales
familycounts <- cbind(namesbytitle %>% as.data.frame,familytotal,familymales,familyfemales,familychildren,familyadults,family_kids_to_adults)

#with Embarked
#familytotal <- namesbytitle[,-1][,3:6]  %>% as.data.frame %>% apply(1,sum)
#familymales <-  namesbytitle[,-1][,c(3,5)] %>% as.data.frame %>% apply(1,sum)
#familyfemales <- namesbytitle[,-1][,c(4,6)] %>% as.data.frame %>% apply(1,sum)
#familychildren <- namesbytitle[,-1][,3:4] %>% as.data.frame %>% apply(1,sum)
#familyadults <- namesbytitle[,-1][,5:6] %>% as.data.frame %>% apply(1,sum)
#familycounts <- cbind(namesbytitle %>% as.data.frame,familytotal,familymales,familyfemales,familychildren,familyadults)

# get stats of numerics Age and Fare
maxrm <- function(x) max(x,na.rm=T)
minrm <- function(x) min(x,na.rm=T)
medrm <- function(x) median(x,na.rm=T)
rangerm <- function(x) max(x,na.rm=T) - min(x,na.rm=T)
familysumm <- grouptt8 %>% summarise_at(.funs=funs(maxrm,minrm,medrm,rangerm),.vars=c("Age","Fare"))

#impute missing values
#famSumm <- familysumm
#familysummImputed <- familysumm[,-(1:2)] %>% sapply(function(x) {x[which(is.na(x) | abs(x)==Inf)] <- median(x,na.rm=T); x})
#famSumm[,-(1:2)] <- familysummImputed  

#with Embarked
#familysummImputed <- familysumm[,-(1:3)] %>% sapply(function(x) {x[which(is.na(x) | abs(x)==Inf)] <- median(x,na.rm=T); x})
#famSumm[,-(1:3)] <- familysummImputed  

#combine the two many-to-one summary tables
#family <- merge(familycounts,famSumm)
family <- merge(familycounts,familysumm)


#impute missing values
fam <- family
familyImputed <- family[,-(1:2)] %>% sapply(function(x) {x[which(is.na(x) | abs(x)==Inf)] <- median(x,na.rm=T); x})
fam[,-(1:2)] <- familyImputed  
family <- fam



#merge many-to-one data with original data
testtrain8 <- merge(testtrain8,family)


trainnames8 <- c(trainnames6,names(family[-(1:2)]))

#with embarked
#trainnames8 <- c(trainnames6,names(family[-(1:3)]))


#impute entire data set, test and training this time

ageind <- which(names(testtrain8)=="Age") #use this to find the Y value
agenameind <- which(trainnames6=="Age") #use this to exclude Age from imputation predictors
ageprednames <- trainnames6[-agenameind]
AgeFareNotMissing <- !is.na(testtrain8$Age) & !is.na(testtrain8$Fare)
Xage8 <- testtrain8[AgeFareNotMissing,][,ageprednames]
# create a "Y" target set: all targets 
Yage <- testtrain8[AgeFareNotMissing,ageind]
RFage <- randomForest(Xage8,Yage,do.trace=100,importance=T,ntree=2000)
agemissing8 <- is.na(testtrain8$Age)
#Xage and X7[,-3] (remove the age column) have the same columns
AgeImpute <- predict(RFage,testtrain8[agemissing8,ageprednames])
varImpPlot(RFage)
testtrain8[agemissing8,"Age"] <- AgeImpute

train8 <- testtrain8[testtrain8$Survived!=-2000,]
X8 <-  train8[,trainnames8]
Y2 <- train8[,"Survived"]
summary(X8,20)
X8keepmissing <- X8

Run8 <- runmodels(X8,Y2,runinteractions = T)


save.image("step10save.RData")

# prepare test data for prediction
test8 <- testtrain8[testtrain8$Survived==-2000,]
Xtest8 <-  test8[,trainnames8]

#preds <- predict()

######## END STEP 10 ###############################################
