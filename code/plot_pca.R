setwd('/Users/jinrui.liu/Desktop/tanseyLab/transfer_learning/transfer-learning/');rm(list=ls())
suppressMessages(require(ggplot2))
suppressMessages(require(dplyr))
suppressMessages(require(reshape2))

target='GDSC'

data_in=read.csv('data/drug/rep-gdsc-ctd2-mean-log.csv')
data_in=data_in[,c('Drug.Name','ccle','drug_id','sample_id',
                   colnames(data_in)[grep(paste0('log_',target,'_published_auc_mean'),colnames(data_in))])]

id_cols=data_in[,c('Drug.Name','ccle','drug_id','sample_id')]

data_out_train=read.csv('results/2023-10-03/test_transfer/train_predictions.csv')
colnames(data_out_train)[2]='sample_id'
data_out_train=left_join(data_out_train,id_cols,by=c('sample_id','drug_id'))

data_out_test=read.csv('results/2023-10-03/test_transfer/test_predictions.csv')
colnames(data_out_test)[2]='sample_id'
data_out_test=left_join(data_out_test,id_cols,by=c('sample_id','drug_id'))

data_out=rbind(data_out_train,data_out_test);rownames(data_out)=paste0(data_out$drug_id,'_',data_out$sample_id)
data_out=data_out[paste0(data_in$drug_id,'_',data_in$sample_id),]
data_out=data_out[,c('Drug.Name','ccle','drug_id','sample_id','predictions')]

identical(paste0(data_out$drug_id,'_',data_out$sample_id),paste0(data_in$drug_id,'_',data_in$sample_id))

cor.test(data_in$log_GDSC_published_auc_mean, data_out$predictions, method = "pearson")

df_in=dcast(data_in,sample_id ~ drug_id,value.var = paste0('log_',target,'_published_auc_mean'))
df_out=dcast(data_out,sample_id ~ drug_id,value.var = 'predictions')


