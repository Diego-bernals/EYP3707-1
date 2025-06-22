
set.seed(2024)
path <- 'https://raw.githubusercontent.com/guru99-edu/R-Programming/master/titanic_data.csv'
titanic <-read.csv(path)
head(titanic)


tail(titanic)

shuffle_index <- sample(1:nrow(titanic))
head(shuffle_index)


titanic <- titanic[shuffle_index, ]
head(titanic)
library(magrittr)
library(dplyr)
# Drop variables
clean_titanic <- titanic %>%
  select(-c(home.dest, cabin, name, x, ticket)) %>% 
  #Convert to factor level
  mutate(pclass = factor(pclass, levels = c(1, 2, 3), labels = c('Upper', 'Middle', 'Lower')), 
         survived = factor(survived, levels = c(0, 1), labels = c('No', 'Yes'))) %>%
  na.omit()

clean_titanic$age<-as.numeric(clean_titanic$age)
clean_titanic$fare<-as.numeric(clean_titanic$fare)
clean_titanic<-na.omit(clean_titanic)

glimpse(clean_titanic)

