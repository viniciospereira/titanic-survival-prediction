######## STEP 11 : PCA ###################################################

#train$Survived  <- as.factor(train$Survived)
#ggpairs(train,columns=(1:12)[-c(1,4,9,11)],mapping=ggplot2::aes(colour=Survived,fill=Survived,alpha=0.5))

#too many dimensions
#trainXY <- cbind(X8[,-17],Y2=Y2 %>% as.factor)
#ggpairs(trainXY,mapping=ggplot2::aes(colour=Y2,fill=Y2,alpha=0.5))


############  PCA
ncolumns <- 12

numonly <- function(X) { isnum <- sapply(X,function (x) is.numeric(x) &  length(unique(x))>2 ); X[,isnum]  }

str(numonly(X8))

mypca <- princomp(X8 %>% numonly %>% convelem ,cor=T)

plot(mypca)

myscores <- cbind(as.data.frame(mypca$scores),Y2=as.factor(Y2))
#ggpairs(myscores,columns= 1:ncolumns,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))

ggpairs(cbind(X8 %>% numonly,Y2=as.factor(Y2)) ,columns= 1:10,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))

ggpairs(myscores,columns= 1:10,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))


### ranking transforms

X8rank <- sapply(X8 %>% numonly %>% convelem,rank)
rankpca <- princomp(X8rank,cor=T)
rankscores <- cbind(as.data.frame(rankpca$scores),Y2=as.factor(Y2))
ggpairs(rankscores,columns= 1:ncolumns,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))


invrankscores <- cbind(apply(mypca$scores,2,rank) %>% as.data.frame,Y2=as.factor(Y2))
ggpairs(invrankscores,columns= 1:ncolumns,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))


#sigmoid transform

logistic <- function(x)  1/(1+exp(-x))

x <- (-100:100)/10
plot(x,logistic(x))

logiscores <- cbind(apply(mypca$scores,2,logistic) %>% as.data.frame,Y2=as.factor(Y2))
ggpairs(logiscores,columns= 1:ncolumns,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))

                          
                          
#is training the same as test ?
#needs work !

tt <- testtrain6
XT2 <- tt %>% select(trainnames6) %>%  numonly %>% ageimputeMedian
YT2 <- ifelse(tt$Survived==-2000,1,0) %>% as.factor
mypcaT <- princomp(XT2,cor=T)
myscoresT <- cbind(as.data.frame(mypcaT$scores),YT2=as.factor(YT2))
ggpairs(myscoresT,columns= 1:ncolumns,mapping=ggplot2::aes(color=YT2,fill=YT2,alpha=0.1))


#### let's use them to predict
myX <- list(basic=X8,
            PCA=myscores[,-ncol(rankscores)],
            rankpca=rankscores[,-ncol(rankscores)],
            invrankpca=invrankscores[,-ncol(rankscores)],
            logistic=logiscores[,-ncol(rankscores)],
            basicPlusPCA=cbind(X8,myscores[,-ncol(rankscores)]),
            basicPlusRankPCA=cbind(X8,rankscores[,-ncol(rankscores)]),
            basicPlusInvRankPCA=cbind(X8,invrankscores[,-ncol(rankscores)]), 
            basicPlusLogistic=cbind(X8,logiscores[,-ncol(rankscores)]), 
            firstPCA=cbind(X8,myscores[,1]),
            firstTwoPCA=cbind(X8,myscores[,1:2]),
            firstFivePCA=cbind(X8,myscores[,1:5]),
            firstTenPCA=cbind(X8,myscores[,1:10]),
            firstInvPCA=cbind(X8,invrankscores[,1]),
            firstTWoInvPCA=cbind(X8,invrankscores[,1:2]),
            firstFiveInvPCA=cbind(X8,invrankscores[,1:5]),
            firstTeninvPCA=cbind(X8,invrankscores[,1:10]),
            firstRankPCA=cbind(X8,rankscores[,1]),
            firstTWoRankPCA=cbind(X8,rankscores[,1:2]),
            firstFiveRankPCA=cbind(X8,rankscores[,1:5]),
            firstTenRankPCA=cbind(X8,rankscores[,1:10]),
            firstLogistic=cbind(X8,logiscores[,1]),
            firstTWoLogistic=cbind(X8,logiscores[,1:2]),
            firstFiveLogistic=cbind(X8,logiscores[,1:5]),
            firstTenLogistic=cbind(X8,logiscores[,1:10]))

            

# xgprep((myX[[i]]  %>% convelem ,Y2)

nsample <- 10
#resmat <- matrix(0,nsample,length(myX))
resmat <- matrix(NA,nsample,length(myX))
for(j in 1:nsample)
{  
  for(i in 1:length(myX) )
  {
    print(names(myX)[i]) 
    RF1 <- randomForest(myX[[i]],as.factor(Y2),do.trace=F,importance=T)
    #RF1 <- randomForest(myX[[i]],as.factor(Y2),do.trace=100,importance=T)
    Yhat <- RF1$votes[,2]
    AUC <- auc(Y2,Yhat)
    resmat[j,i] <- AUC
    print(c("auc :",AUC))  

  
    #xgb_hist = buildXGBOOST(myX[[i]]  %>% convelem ,Y2,verb=F)
    #print(xgb_hist$xgbauc)
   }
   #names(resmat) <- names(myX)
   resdf <- as.data.frame(t(resmat))
   row.names(resdf) <- names(myX)
   resdf2 <- t(resdf) %>% as.data.frame
   names(resdf2) <- names(myX)
   gresdf <- resdf2 %>% gather(key="dataset",value="AUC")
   #a <- ggplot(gresdf, aes(y=AUC, x=as.factor(dataset))) + geom_boxplot(aes(fill=as.factor(dataset)), stat="boxplot", position="dodge", alpha=0.5, width=0.2) + geom_violin(aes(fill=as.factor(dataset)), stat="ydensity", position="dodge", alpha=0.5, trim=TRUE, scale="area") + coord_flip() + theme_grey() + theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + guides(fill=guide_legend(title="dataset")) + xlab("as.factor(dataset)") + ylab("AUC")

   a <- ggplot(gresdf[with(gresdf, dataset %in% c("basic", "basicPlusInvRankPCA", "basicPlusLogistic", "basicPlusPCA", "basicPlusRankPCA", "firstFiveInvPCA", "firstFiveLogistic", "firstFivePCA", "firstFiveRankPCA", "firstInvPCA", "firstLogistic", "firstPCA", "firstRankPCA", "firstTeninvPCA", "firstTenLogistic", "firstTenPCA", "firstTenRankPCA", "firstTWoInvPCA", "firstTWoLogistic", "firstTwoPCA", "firstTWoRankPCA")), ], aes(y=AUC, x=as.factor(dataset))) + geom_boxplot(aes(fill=as.factor(dataset)), stat="boxplot", position="dodge", alpha=0.5, width=0.2) + geom_violin(aes(fill=as.factor(dataset)), stat="ydensity", position="dodge", alpha=0.5, trim=TRUE, scale="area") + coord_flip() + theme_grey() + theme(text=element_text(family="sans", face="plain", color="#000000", size=15, hjust=0.5, vjust=0.5)) + guides(fill=guide_legend(title="dataset")) + xlab("as.factor(dataset)") + ylab("AUC")
   print(a)
   #matplot(resdf[c(1,5:15),1:j])
}


row.names(resdf) <- names(myX)
resdf2 <- t(resdf) %>% as.data.frame
names(resdf2) <- names(myX)
gresdf <- resdf2 %>% gather(key="dataset",value="AUC")
ggraptR(gresdf)



svd8 <- svd(X8 %>% convelem)

X8svd <- cbind(X8,svd8$u)

svd8 <- svd(X8 %>% convelem)
svd8invrank <- svd8$u %>% as.data.frame %>% sapply(rank) %>% as.data.frame
svd8rank <- svd(X8 %>% convelem %>% sapply(rank))

svd_scores <- cbind(svd8$u %>% as.data.frame,Y2=as.factor(Y2))
svd_rank_scores <- cbind(svd8rank$u %>% as.data.frame,Y2=as.factor(Y2))
svd_invrank_scores <- cbind(svd8invrank %>% as.data.frame,Y2=as.factor(Y2))
ggpairs(svd_scores,columns= 1:7,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))
ggpairs(svd_rank_scores,columns= 1:7,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))
ggpairs(svd_invrank_scores,columns= 1:7,mapping=ggplot2::aes(color=Y2,fill=Y2,alpha=0.5))


RFsvd <- randomForest(X8svd,as.factor(Y2),do.trace=100,importance=T)
print(auc(Y2,RFsvd$votes[,2]))

RFsvd2 <- randomForest(svd8$u,as.factor(Y2),do.trace=100,importance=T)
print(auc(Y2,RFsvd2$votes[,2]))

RFsvdInvRank <- randomForest(cbind(X8,svd8invrank),as.factor(Y2),do.trace=100,importance=T)
print(auc(Y2,RFsvdInvRank$votes[,2]))
RFsvdInvRank <- randomForest(cbind(X8,svd8invrank[,1:7]),as.factor(Y2),do.trace=100,importance=T)
print(auc(Y2,RFsvdInvRank$votes[,2]))



save.image("step11save.RData")


############ END STEP 11 #########################
