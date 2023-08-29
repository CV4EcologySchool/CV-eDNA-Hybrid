library(jsonlite)
image_multilab = lapply(split(invertmatch$LKTL_Long, invertmatch$Event), unique)
dna_multilab = lapply(split(dna_LKTL_Long, edna_rmdup$event), unique)

imageJSON = toJSON(image_multilab)
dnaJSON = toJSON(dna_multilab)

write(imageJSON, "image_multilab.json")
write(dnaJSON, "dna_multilab.json")
