
keans number of cluster-------------------------------------------------------------------------

install.packages("factoextra")
install.packages("cluster")
library(factoextra)
library(cluster)
df=USAccDeaths
df=na.omit(df)
df=scale(df)
head(df)
fviz_nbclust(df,kmeans,method="wss")
km <- kmeans(df,center=4,nstart = 25)
km
fviz_cluster(km,data = df)


NCR--------------------------------------------------------------------------------------------------

factorial<-function(val)
{
  fact=1
  if(val<0){
    return(0)
  }else if(val==0){
    return(1)
  }else{
    for(i in 1:val){
      fact=fact*i
    }
    return(fact)
  }
}
n<-as.integer(readline(prompt = "enter n value"))
r<-as.integer(readline(prompt = "enter r value"))
c=n-r
nfact=factorial(n)
rfact=factorial(r)
nrfact=factorial(c)
ans=nfact/(rfact*nrfact)
print("ncr value is")
print(ans)



binary--------------------------------------------------------------------------------------------------------

bisearch = function(table,key){
  stopifnot(is.vector(table),is.numeric(table))
  r = length(table)
  m = ceiling(r/2L)
  if(table[m]>key){
    if(r==1L){
      return(FALSE)
    }
    bisearch(table[1L:(m-1L)],key)
  }
  else if(table[m]<key){
    if(r == 1L){
      return(FALSE)
    }
    bisearch(table[(m+1L):r],key)
  }
  else{
    return(TRUE)
  }
  }
n<-as.integer(readline(prompt = "enter n value"))
bisearch(c(1,2,3,4,5,6),n)
  
  
  import export xlsx-----------------------------------------------------------------------------------------
  #import excel
install.packages("readxl")
library(readxl)
exp1 <- read_excel("D://sudharson//exp2.xlsx")
exp1 <- read.csv("D://sudharson//exp66.csv")
View(exp1)

#export excel
install.packages("openxlsx")
library(openxlsx)
s.no <- seq(1,3,by=1)
fruits <- c("apple","mango","orange")
shop1 <-c(12,14,16)
shop2 <- c(22,15,24)
shop3 <- c(12,15,14)
shop4 <-c(66,26,33)
rate <- data.frame(fruits,shop1,shop2,shop3,shop4)
rate
write.xlsx(rate,file="D://sudharson//exp0.xlsx")
write.csv(rate,file="D://sudharson//exp66.csv")


fibonacci series--------------------------------------------------------------------------------------
nterms <- as.integer(readline(prompt="How many terms? ")) 
n1 = 0
n2 = 1
count = 2 if(nterms <= 0) {
print("Please enter a positive integer")

} else {

if(nterms == 1) {
print("Fibonacci sequence:")
print(n1)
} else {

print("Fibonacci sequence:")
print(n1)
print(n2)

while(count < nterms) {
nth = n1 + n2
print(nth)
n1 = n2 n2 = nth
count = count + 1

}

}

}



data types-----------------------------------------------------------
#Logical data type
bool1 <- TRUE print(bool1)
print(class(bool1))
bool2 <- FALSE print(bool2)
print(class(bool2))
#Numeric data type
# floating point values
weight <- 63.5 print(weight)
print(class(weight))
# Real numbers
height <- 182 print(height)
print(class(height))
#Integer datatype
integer_variable<- 186L
print(class(integer_variable))
# Complex data type
complex_value<- 3 + 2i
# print class of complex_value
 
print(class(complex_value))
complex_value


#Character data type
# Create a string variable

fruit <- "Apple"
print(class(fruit))
# Create a character variable
my_char<- 'A'
print(class(my_char))
#Raw data type
# convert character to raw

raw_variable<- charToRaw("Welcome to r program")
print(raw_variable)
print(class(raw_variable))
# convert raw to character 
char_variable<- rawToChar(raw_variable)
print(char_variable)
print(class(char_variable))
# Create a vector.
apple <- c('red','green',"yellow")
print(apple)
# Get the class of the vector.
print(class(apple))
# Create a list.
list1 <- list(c(2,5,3),21.3,sin)
# Print the list.
print(list1)
# Create a matrix.
M = matrix( c('a','a','b','c','b','a'), nrow = 2, ncol = 3, byrow = TRUE)
print(M)
# Create an array.
a <- array(c('green','yellow'),dim = c(3,3,2))
print(a)
 
# Create a vector.

apple_colors<- c('green','green','yellow','red','red','red','green')

# Create a factor object.

factor_apple<- factor(apple_colors)

# Print the factor.
print(factor_apple)
print(nlevels(factor_apple))
# Create the data frame.
BMI < - data.frame(
gender = c("Male", "Male","Female"), height = c(152, 171.5, 165),
weight = c(81,93, 78), Age = c(42,38,26)
)
print(BMI)



clustering------------------------------------------------------------------------

install.packages("Stats")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("ggfortify")
library(stats)
library(dplyr)
library(ggplot2)
library(ggfortify)
View(iris)
mydata = select(iris,c(1,2,3,4))
wssplot<- function(data, nc=15, seed=1234)
{
wss <-(nrow(data-1))*sum(apply(data,2,var))
for(i in 2:nc){
set.seed(seed)
wss[i] <- sum(kmeans(data,centers=i)$withinnss)}
plot(1:nc,wss,type="b", xlab="number of clusters",
ylab="within groups sum of squares")
}
wssplot(mydata)
KM = kmeans(mydata,2)
autoplot(KM,mydata,frame=TRUE)



ariori-----------------------------------------------------------------------------

install.packages("arules")
install.packages("arulesViz")
library(arules)
library(arulesViz)
data("Groceries")
summary(Groceries)
apriori(Groceries,parameter=list(support=0.002,confidence=0.5))->rule1
inspect(head(rule1,10))
inspect(head(sort(rule1,by="lift"),5))
plot(rule1)
plot(rule1,method="grouped")
apriori(Groceries,parameter=list(support=0.002,confidence=0.5,minlen=3))->rule2
inspect(head(rule2,7))
plot(rule2)
plot(rule2,method="grouped")
apriori(Groceries,parameter=list(support=0.007,confidence=0.6))->rule3
inspect(head(rule3,4))
plot(rule3,method="grouped")


linear---------------------------------------------------------

a<-c(151,174,138,186,128,136,179,163,152,131)
b<-c(63,81,56,91,47,57,76,72,62,48)
relation<-lm(b~a)
a<-data.frame(a=170)
result<-predict(relation,a)
print(result)
install.packages('readxl')
library('readxl')
ads<-read.csv('d:/advertising.csv')
View(ads)
nrow(ads)
ncol(ads)
colnames(ads)
Tv<-ads$TV
Sales<-ads$Sales
plot(Tv,Sales)
model<-lm(Sales~Tv)
summary(model)
attributes(model)
coefficients(model)
coef(model)
abline(model)


logistic---------------------------------------------------------------


sex<-puffinbill$sex
curlen<-puffinbill$curlen
sexcode<-ifelse(sex == "F",1,0)
plot(curlen, jitter(sexcode, 0.15), pch = 19, xlab = "Bill length (mm)", ylab = "Sex (0 - male, 1 - female)")
model<- glm(sexcode~curlen, binomial)
summary(model)
xv<- seq(min(curlen),max(curlen),0.01)
yv<- predict(model,list(curlen=xv),type="response")
lines(xv, yv, col = "red")
library(popbio)
logi.hist.plot(curlen,sexcode,boxp = FALSE,type= "count",col="gray",xlabel = "size")



navie Bayes------------------------------------------------------------

install.packages("e1071")
install.packages("caTools")
install.packages("caret")
library(e1071)
library(caTools)
library(caret)
library(ggplot2)
split <- sample.split(iris, SplitRatio = 0.7)
train_cl <- subset(iris, split == "TRUE")
test_cl <- subset(iris, split == "FALSE")
train_scale <- scale(train_cl[, 1:4])
test_scale <- scale(test_cl[, 1:4])
set.seed(120) # Setting Seed
classifier_cl <- naiveBayes(Species ~ ., data = train_cl)
classifier_cl
y_pred <- predict(classifier_cl, newdata = test_cl)
cm <- table(test_cl$Species, y_pred)
cm
confusionMatrix(cm)

decision tree-----------------------------------------------------------------

install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
data = read.csv("F:\\RRR\\gender.csv")
tree <- rpart(Height ~ Gender+Weight,data)
a <- data.frame(Gender=c("Female"),Weight=c(76))
result <- predict(tree,a)
print(result)
rpart.plot(tree)
tree <- rpart(Gender ~ Height+Weight,data)
a <- data.frame(Height=c(149),Weight=c(75))
result <- predict(tree,a)
print(result)
rpart.plot(tree)

time series arima model-------------------------------------------------------------------------

install.packages('forecast')
library(forecast)
weather_prod_input <- as.data.frame( read.csv("weather.csv") )
weather_prod <- ts(weather_prod_input[,3])
plot(weather_prod, xlab = "Time (months)",
ylab = "weather between(1901-2017)")
plot(diff(weather_prod))
abline(a=0, b=0)
acf(diff(weather_prod), xaxp = c(0, 48, 4), lag.max=48, main="")
pacf(diff(weather_prod), xaxp = c(0, 48, 4), lag.max=48, main="")
arima_1 <- arima (weather_prod,
order=c(0,1,0),
seasonal = list(order=c(1,0,0),period=12))
arima_1
acf(arima_1$residuals, xaxp = c(0, 48, 4), lag.max=48, main="")
pacf(arima_1$residuals, xaxp = c(0, 48, 4), lag.max=48, main="")
arima_2 <- arima (weather_prod,
order=c(0,1,1),
seasonal = list(order=c(1,0,0),period=12))
arima_2
acf(arima_2$residuals, xaxp = c(0, 48, 4), lag.max=48, main="")
pacf(arima_2$residuals, xaxp = c(0, 48,4), lag.max=48, main="")
#Normality and Constant Variance
plot(arima_2$residuals, ylab = "Residuals")
abline(a=0, b=0)
hist(arima_2$residuals, xlab="Residuals", xlim=c(-20,20))

qqnorm(arima_2$residuals, main="")
qqline(arima_2$residuals)
#Forecasting
#predict the next 12 months
?predict()
arima_2.predict <- predict(arima_2,n.ahead=12)
?matrix()
matrix(c(arima_2.predict$pred-1.96*arima_2.predict$se,
arima_2.predict$pred,
arima_2.predict$pred+1.96*arima_2.predict$se), 12,1,
dimnames=list( c(117:128) ,c("Pred")) )
plot(weather_prod)
lines(arima_2.predict$pred)
lines(arima_2.predict$pred+1.96*arima_2.predict$se, col=4, lty=2)
lines(arima_2.predict$pred-1.96*arima_2.predict$se, col=4, lty=2)
?arima()



test analysis--------------------------------------------------------------------------------


#Term Frequency(TF)
tf<-function(row){
return(row/sum(row))
}
#Inverse Document Frequency(IDF)
idf<-function(col){
return(log10(length(col)/length(which(col>0))))
}
#Term Frequency - Inverse Document Frequency(TF-IDF)
tfidf<-function(token.matrix){
tf<-t(apply(token.matrix,1,tf))
idf<-apply(token.matrix,2,idf)
return(tf*idf)
}
#Testing the Function
install.packages("tm")
library(tm)
text<-c("A dog and a cat","The dog is barking","The cat is on the wall")
corp<-Corpus(VectorSource(text))
#Finding the tokens
doc.tokens<-as.matrix(DocumentTermMatrix(corp))
tfidf(doc.tokens)



sql essestial----------------------------------------------------------------

install.packages("sqldf")
install.packages("readr")
library(sqldf)
library(readr)
Book1<-read_csv("Book1.csv")
Book2<-read_csv("Book2.csv")
Book3<-read_csv("Book3.csv")
View(Book1)
View(Book2)
View(Book3)
#Set_Operations
sqldf("SELECT * FROM Book1 UNION SELECT * FROM Book2")
sqldf("SELECT * FROM Book1 UNION ALL SELECT * FROM Book2")
sqldf("SELECT * FROM Book1 INTERSECT SELECT * FROM Book2")
sqldf("SELECT * FROM Book1 EXCEPT SELECT * FROM Book2")
#Join_Operations
sqldf("SELECT Book1.Name,Book3.College FROM Book1 INNER JOIN
Book3 ON Book1.Id=Book3.Id")
sqldf("SELECT Book1.Name,Book3.College FROM Book1 FULL OUTER
JOIN Book3 ON Book1.Id=Book3.Id")
sqldf("SELECT Book1.Name,Book3.College FROM Book1 LEFT OUTER
JOIN Book3 ON Book1.Id=Book3.Id")
sqldf("SELECT Book1.Name,Book3.College FROM Book1 RIGHT OUTER
JOIN Book3 ON Book1.Id=Book3.Id")
#Grouping_Extensions
Book4<- read_csv("Book4.csv")
View(Book4)
sqldf("SELECT Department,SUM(Salary) as Salary FROM Book4 GROUP BY
Department")
sqldf("SELECT Department,Category,SUM(Salary) as Salary FROM Book4
GROUP BY Department, Category")
sqldf("SELECT Department,SUM(Salary) as Salary FROM Book4 GROUP BY
Department HAVING SUM(Salary) = 25000")
sqldf("SELECT Department,Category,SUM(Salary) as Salary FROM Book4
GROUP BY Department, Category HAVING SUM(salary) = 50000")



sentimental analysis--------------------------------------------------------------------------------



install.packages("tm")
install.packages("wordcloud")
install.packages("syuzhet")
library(tm)
library(wordcloud)
library(syuzhet)
reviews<-read.csv(file.choose(),header=1)
str(reviews)
corpus<-iconv(reviews$text)
corpus<-Corpus(VectorSource(corpus))
inspect(corpus[1:10])
corpus <- tm_map(corpus,tolower)
corpus <- tm_map(corpus,removePunctuation)
corpus <- tm_map(corpus,removeNumbers)
corpus <- tm_map(corpus,removeWords,stopwords("english"))
#corpus <- tm_map(corpus,removeWords,c("book","read","life"))
corpus <- tm_map(corpus,stripWhitespace)
inspect(corpus[1:10])
reviews_final <- corpus
tdm <- TermDocumentMatrix(reviews_final)
tdm <- as.matrix(tdm)
tdm[1:10,1:10]
w <- sort(rowSums(tdm),decreasing = T)
set.seed(2000)
wordcloud(words = names(w),
freq = w,
max.words=50,
random.order = T,
min.freq=5,
colors=brewer.pal(25,"Dark2"),
scale = c(5,0.5))
sentiment_data <- iconv(reviews$text)
s <- get_nrc_sentiment(sentiment_data)
s[1:10,]
s$score <- s$positive - s$negative
s[1:10,]
write.csv(x=s,file="reviews.csv")
review_score <- colSums(s[,])
print(review_score)
barplot(colSums(s),
las =2,
col=rainbow(10),
ylab='Count',
main='Sentiment')






advanced sql------------------------------------------------------------------------


create table employees1(id int,name varchar(30),salary int,age int,location
varchar(30),mobile int);
insert into employees1 values(101,'Suka',50000,22,'Palayamkottai',8667897130);
insert into employees1 values(102,'Saran',40000,25,'Tuticorin',9807654321);
insert into employees1 values(103,'Priya',30000,29,'Tirunelveli',8765432190);
insert into employees1 values(104,'Hari',20000,24,'Chennai',9678543210);
insert into employees1 values(105,'Pavi',36000,23,'Tenkasi',9390876543);
insert into employees1 values(106,'Ram',55000,25,'Kanyakumari',9345678902);
select * from employees1;
select count(age) from employees1;
select AVG(salary) from employees1;
select SUM(salary) from employees1;
select salary from employees1 where id=104;
select min(salary) from employees1;
select max(salary) from employees1;
create table students11(id int,name varchar(20),subject_name
varchar(20),Marks_scored int,Total_marks int);
insert into students11 values(1,'Joseph','DIP',99,100);
insert into students11 values(2,'Ravi','IOT',90,100);
insert into students11 values(3,'Jack','DS',87,100);
insert into students11 values(4,'Steve','SC',85,100);
insert into students11 values(5,'Karthi','Python',95,100);
insert into students11 values(6,'Devi','JAVA',80,100);
select * from students11;
select Marks_scored,id,name,rank() over(order by Marks_scored
desc),dense_rank() over(order by Marks_scored desc),row_number() over(order by
Marks_scored desc)from students11;
with cte as(select AVG(Marks_scored) as average_marks from students11)
select students11.*,cte.average_marks from students11,cte
with cte as(select MIN(Marks_scored) as Min_marks from students11)
select students11.*,cte.Min_marks from students11,cte
with cte as(select MAX(Marks_scored) as Max_marks from students11)
select students11.*,cte.Max_marks from students11,cte






