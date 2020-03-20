# Load the data

source("03_load_data.R")

library("data.table")
library("mlr3verse")
library("rsample")
library("skimr")
skim(adult)
# pairs(adult)
DataExplorer::plot_histogram(adult, ncol = 3)
DataExplorer::plot_boxplot(adult, by = "salary", ncol = 3)
DataExplorer::plot_intro(adult)
DataExplorer::plot_correlation(adult)

# To do the basic data analysis
```{r}
adult_tr = adult
adult_tr$salary<-ifelse(adult_tr$salary==" <=50K",0,1)
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2

dmy<-dummyVars("~.",data=adult_tr)
adultsTrsf<-data.frame(predict(dmy,newdata=adult_tr))
dim(adult_tr)
dim(adultsTrsf)
head(adultsTrsf)
str(adultsTrsf)
cor.prob<-function(X,dfr=nrow(X)-2){
  R<-cor(X,use="pairwise.complete.obs")
  above<-row(R)<col(R)
  r2<-R[above]^2
  Fstat<-r2*dfr/(1-r2)
  R[above]<-1-pf(Fstat,1,dfr)
  R[row(R)==col(R)]<-NA
  R
}

flattenSquareMatrix<-function(m){

  if((class(m) !="matrix") | (nrow(m) !=ncol(m))) stop("Must be asquare matrix.")
  if(!identical(rownames(m),colnames(m))) stop("Row and column names must be equal.")
  ut<-upper.tri(m)
  data.frame(i=rownames(m)[row(m)[ut]],
             j=rownames(m)[col(m)[ut]],
             cor=t(m)[ut],
             p=m[ut])
}
corMasterList<-flattenSquareMatrix(cor.prob(adultsTrsf))

dim(corMasterList)
corList<-corMasterList[order(-abs(corMasterList$cor)),]
head(corList)
selectedSub<-subset(corList,(abs(cor)>0.2 & j =="salary"))
bestSub<-as.character(selectedSub$i[c(1,3,5,6,8,9)])
library(psych)
pairs.panels(adultsTrsf[c(bestSub,"salary")])
