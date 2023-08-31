library(jsonlite)
image_multilab = lapply(split(train_split1$LKTL_Long, train_split1$Event), unique)
dna_multilab = lapply(split(dna_train_LKTL_Long, dna_train$event), unique)

imageJSON = toJSON(image_multilab)
dnaJSON = toJSON(dna_multilab)

write(imageJSON, "image_multilab.json")
write(dnaJSON, "dna_multilab.json")
