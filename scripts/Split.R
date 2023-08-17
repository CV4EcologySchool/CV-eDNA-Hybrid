#See Markdown file for additional details
library(dplyr)

invert = read.csv("Invert_Meta_Final.csv")

'%!in%' = function(x,y)!('%in%'(x,y))

abund_by_site_LKTL <- invertmatch %>%
  group_by(LKTL, Site) %>%
  summarise(abundance = n())

result <- abund_by_site_LKTL %>%
  left_join(LKTL_Table, by = "LKTL") %>%
  mutate(normalized_abundance = abundance / Freq)

filtered_sites <- result %>%
  group_by(Site) %>%
  filter(all(normalized_abundance > 0 & normalized_abundance < 0.25))

num_samples <- nrow(invertmatch)
num_train <- round(0.75 * num_samples)
num_validation <- round(0.15 * num_samples)
num_test <- num_samples - num_train - num_validation

# Shuffle the indices
shuffled_indices <- sample(num_samples)

invertmatch$Split = NA
invertmatch$Split[shuffled_indices[1:num_train]] = "Train"
invertmatch$Split[shuffled_indices[(num_train + 1):(num_train + num_validation)]] = 'Validation'
invertmatch$Split[shuffled_indices[(num_train + num_validation + 1):num_samples]] = "Test"




alltable = table(invert$AllTaxa)
allname = names(which(alltable <= 99))

#Train:Test ratio is 7:3, so I am initializing those values here
fractionTraining = 0.70
fractiontest = 0.30

seeds = 2833

set.seed(seeds)

# Compute sample sizes.
sampleSizeTraining = floor(fractionTraining * nrow(invert))
sampleSizetest = floor(fractiontest * nrow(invert))
# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining = sort(sample(seq_len(nrow(invert)), size=sampleSizeTraining))
indicestest = setdiff(seq_len(nrow(invert)), indicesTraining)

# Finally, output the three dataframes for training, testing, and zero shot.
dfTraining = invert[indicesTraining, ]
dftest = invert[indicestest, ]

set = subset(dfTraining, AllTaxa %!in% allname)
alltrain = set

set = subset(dftest, AllTaxa %!in% allname)
alltest = set

allzero = subset(invert, AllTaxa %in% allname)
# write.csv(alltest, "test.csv", row.names = F)
# write.csv(alltrain, "train.csv", row.names = F)
# write.csv(allzero, "zero.csv", row.names = F)


