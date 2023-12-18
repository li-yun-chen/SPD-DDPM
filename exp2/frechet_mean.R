library(shapes)



n <- 1100 
num <- 20 
m <- 10  
data <- read.csv("data/condition/ddpm_spds_list.csv")
result <- matrix(0,n,m*m)

for(i in c(1:1100)){
  sample <- data[((i-1)*num+1):(i*num),]
  samples_without_nan <- sample[complete.cases(sample), ]
  n_sample <- dim(samples_without_nan)[1]
  
  if(n_sample < 2){
    mean <- matrix(NA, nrow = 1, ncol = 100)
    result[i,] <- mean
    next
  }
  
  sample_array <- array(0,c(m,m,n_sample) )
  for(j in c(1:n_sample)){
    tem <- as.matrix(samples_without_nan[j,],m,m)
    sample_array[,,j] <- tem
  }
  print(i)
  mean <- as.vector(estcov(sample_array , method="Riemannian",weights=1)$mean) 
  result[i,] <- mean
}


write.csv(result, file = "data/condition/data_2.csv", row.names = FALSE)










