library(class)
library(gmodels)
library(caret)
library(MLmetrics)

iris_data <- read.csv("iris_data.csv", stringsAsFactors = FALSE)
names(iris_data) <- c("Sepal.Length","Sepal.Width","Petal.Length",
                 "Petal.Width","Species")

table(iris_data$Species)

iris_data$Species <- as.factor(iris_data$Species)
str(iris_data)

round(prop.table(table(iris_data$Species)) * 100, digits = 1)

summary(iris_data[,names(iris_data) != "Species"])

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

iris_data_n <- as.data.frame(lapply(iris_data[1:4], normalize))

summary(iris_data_n$Sepal.Length)

## 80% of the sample size
smp_size <- floor(0.8 * nrow(iris_data_n))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(iris_data_n)), size = smp_size)

iris_train <- iris_data_n[train_ind, ]
iris_test <- iris_data_n[-train_ind, ]
iris_train_labels <- iris_data[train_ind, 5]
iris_test_labels <- iris_data[-train_ind, 5]

trControl <- trainControl(method  = "cv",
                          number  = 5)

model1 <- train(iris_train,
             iris_train_labels,
             method     = "rf",
             metric = "Accuracy",
             trControl  = trControl)

plot(model1)

iris_test_pred1 <- predict(model1,iris_test)

Accuracy(as.numeric(iris_test_pred1),as.numeric(iris_test_labels))

model2 <- train(iris_train,
                iris_train_labels,
                method     = "knn",
                metric = "Accuracy",
                tuneGrid = expand.grid(k=c(1:20)),
                trControl  = trControl)

plot(model2)

iris_test_pred2 <- predict(model2,iris_test)

Accuracy(as.numeric(iris_test_pred2),as.numeric(iris_test_labels))

model3 <- train(iris_train,
                iris_train_labels,
                method     = "xgbLinear",
                metric = "Accuracy",
                trControl  = trControl)

plot(model3)

iris_test_pred3 <- predict(model3,iris_test)

Accuracy(as.numeric(iris_test_pred3),as.numeric(iris_test_labels))
