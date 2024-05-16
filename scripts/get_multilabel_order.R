library(jsonlite)
image_multilab = lapply(split(invert_cleanlab$longlab, invert_cleanlab$Event), unique)
dna_multilab = lapply(split(edna_rmdup$longlab, edna_rmdup$Event), unique)

imageJSON = toJSON(image_multilab)
dnaJSON = toJSON(dna_multilab)

write(imageJSON, "image_multilab_order.json")
write(dnaJSON, "dna_multilab_order.json")
