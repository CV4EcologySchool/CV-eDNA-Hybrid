
setwd("C:/Carabid_Data/CV-eDNA/splits/order")

train_noise = train
valid_noise = valid

for(i in 87:103){
  train_noise[,i] = sample(0:1, nrow(train_noise), replace = TRUE)
  valid_noise[,i] = sample(0:1, nrow(valid_noise), replace = TRUE)
}

write.csv(train_noise, "train_noise.csv", row.names = F)
write.csv(valid_noise, "valid_noise.csv", row.names = F)
