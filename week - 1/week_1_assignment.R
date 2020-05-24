x_train = read.csv('Linear_X_Train.csv')
y_train = read.csv('Linear_Y_Train.csv')
x_test = read.csv('Linear_X_Test.csv')


regressor = lm(formula = y_train$y ~ x,
               x_train)
y_pred = predict(regressor,newdata = x_test)



ggplot() +
  geom_line(aes(x = x_test$x, y = predict(regressor,newdata = x_test)),
            color = 'blue') +
  xlab("Amount spent on coding") +
  ylab("Performance achieved") +
  ggtitle("Analysis of study time v/s performance")

write.csv(y_pred,"y_test.csv",row.names = FALSE)


y_pred2 = read.csv('y_test.csv')
colnames(y_pred2)[1] <- c("y_predicted")

write.csv(y_pred2,"y_test.csv",row.names = FALSE)