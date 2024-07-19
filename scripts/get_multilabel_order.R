
library(jsonlite)
image_multilab = lapply(split(invert_cleanlab$longlab, invert_cleanlab$Event), unique)

edna_rmna = edna_rmdup[!is.na(edna_rmdup$order_plus),]
dna_multilab = lapply(split(edna_rmna$longlab, edna_rmna$Event), unique)

imageJSON = toJSON(image_multilab)
dnaJSON = toJSON(dna_multilab)

write(imageJSON, "image_multilab_order.json")
write(dnaJSON, "dna_multilab_order.json")

image_multilab_train = image_multilab[unique(train$Event)]
dna_multilab_train = dna_multilab[unique(train$Event)]

image_trainJSON = toJSON(image_multilab_train)
dna_trainJSON = toJSON(dna_multilab_train)

write(image_trainJSON, "image_multilab_order_train.json")
write(dna_trainJSON, "dna_multilab_order_train.json")
