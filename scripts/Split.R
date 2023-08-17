library(dplyr)

setwd("C:/Users/jarre/ownCloud/CV-eDNA")

invertmatch = read.csv("invertmatch.csv")

'%!in%' = function(x,y)!('%in%'(x,y))

# Exploring abundance by site
# abund_by_site_LKTL <- invertmatch %>%
#   group_by(LKTL, Site) %>%
#   summarise(abundance = n())
# 
## Proportion by site
# result <- abund_by_site_LKTL %>%
#   left_join(LKTL_Table, by = "LKTL") %>%
#   mutate(normalized_abundance = abundance / Freq)
# 
## Trying to find sites that match the criteria in filter
# filtered_sites <- result %>%
#   group_by(Site) %>%
#   filter(all(normalized_abundance > 0 & normalized_abundance < 0.25))

# Defining split sizes
num_samples <- nrow(invertmatch)
num_train <- round(0.75 * num_samples)
num_validation <- round(0.15 * num_samples)
num_test <- num_samples - num_train - num_validation

# Shuffle the indices
shuffled_indices <- sample(num_samples)

# Assigning splits
invertmatch$Split = NA
invertmatch$Split[shuffled_indices[1:num_train]] = "Train"
invertmatch$Split[shuffled_indices[(num_train + 1):(num_train + num_validation)]] = 'Validation'
invertmatch$Split[shuffled_indices[(num_train + num_validation + 1):num_samples]] = "Test"
