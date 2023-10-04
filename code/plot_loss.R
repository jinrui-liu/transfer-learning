setwd('/Users/jinrui.liu/Desktop/tanseyLab/transfer_learning/transfer-learning/');rm(list=ls())
suppressMessages(require(ggplot2))

dplot=read.csv('scratch/losses.csv')

ggplot(data=dplot, aes(x=step, y=loss)) +
  geom_line()

