orders = c("Araneae",
           "Blattodea",
           "Coleoptera",
           "Diptera",
           "Hemiptera",
           "Hymenoptera",
           "Isopoda",
           "Lepidoptera",
           "Opilioacarida",
           "Opiliones",
           "Orthoptera",
           "Zygentoma")

subclass = c("Acari",
             "Collembola")

class = c("Gastropoda")

subphylum = c("Myriapoda")

phylum = c("Annelida")


lvl_dict = list(Order = orders,
                Subclass = subclass,
                Class = class,
                Subphylum = subphylum,
                Phylum = phylum)

old_levels = c()
for(i in 1:length(newclass)){
  taxon = newclass[i]
  old_levels[i] =  names(lvl_dict)[vapply(lvl_dict, function(x) taxon %in% x, logical(1))]
}

spec_change = data.frame(cbind(old_levels, splevels))
colnames(spec_change) = c("Original", "New")

write.csv(spec_change, "ML_DNABias.csv")


