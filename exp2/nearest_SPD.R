library(Matrix)
library(cvms)
library(data.table)

data <- read.csv("data/condition/eud_ddpm_spds_mean.csv")
m <- sqrt(ncol(data))

find_nearest_spd <- function(vector) {
  mat <- matrix(data = unlist(vector), nrow = 10, byrow = TRUE)
  eig <- eigen(mat, symmetric = TRUE)
  ev <- eig$values
  ev[ev < 0] <- 0
  spd_mat <- eig$vectors %*% diag(ev) %*% t(eig$vectors)
  result_vector <- as.vector(spd_mat)
  return(result_vector)
}

spd_matrices <- data.frame(matrix(NA, nrow = 1100, ncol = 100))

for (i in 1:length(matrices)) {
  spd_matrices[i,] <- find_nearest_spd(data[i,])
}

write.csv(spd_matrices, "data/condition/data_3.csv", row.names = FALSE)


















