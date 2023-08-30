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




## Sampling by event

# events = as.data.frame(table(invert_cleanlab$Event))
# 
# shuffled_data <- events[sample(nrow(events)), ]
# 
# # Initialize variables for tracking
# cumulative_freq <- 0
# sampled_events <- character(0)
# stop = nrow(invert_cleanlab) * 0.15
# 
# # Iterate through shuffled rows
# for (i in 1:nrow(shuffled_data)) {
#   site <- as.character(shuffled_data[i, "Var1"])
#   freq <- shuffled_data[i, "Freq"]
#   
#   # Check if adding the current frequency exceeds 100
#   if (cumulative_freq + freq >= stop) {
#     sampled_events <- c(sampled_events, site)
#     break  # Exit loop since condition is satisfied
#   } else {
#     sampled_events <- c(sampled_events, site)
#     cumulative_freq <- cumulative_freq + freq
#   }
# }
# 
# test = invert_cleanlab[which(invert_cleanlab$Event %in% sampled_events),]



# Defining split sizes
num_samples <- nrow(invert_cleanlab)
num_train <- round(0.75 * num_samples)
num_validation <- round(0.15 * num_samples)
num_test <- num_samples - num_train - num_validation

# Shuffle the indices
shuffled_indices <- sample(num_samples)

# Assigning splits
train_data = invert_cleanlab[shuffled_indices[1:num_train],]
validation_data = invert_cleanlab[shuffled_indices[(num_train + 1):(num_train + num_validation)],]
test_data = invert_cleanlab[shuffled_indices[(num_train + num_validation + 1):num_samples],]

write.csv(train_data, "train.csv")
write.csv(validation_data, "valid.csv")
write.csv(test_data, "test.csv")

