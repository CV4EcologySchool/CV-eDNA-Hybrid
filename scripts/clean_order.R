# Initializes invert_cleanlab dataframe. This removes some junk groups like 'Arthropoda'
# It also removes underrepresented Orders while merging some orders with their respective Class
# or Phylum labels (e.g. Annelida)

setwd("C:/Users/jarre/ownCloud/CV-eDNA")

invertmatch = read.csv("invertmatch.csv")
ednamatch = read.csv("ednamatch.csv")
hierarchy = read.csv("hierarchy_order.csv")
edna_rmdup = read.csv("edna_rmdup.csv")


invert_cleanlab = invertmatch[-which(invertmatch$AllTaxa == "Arthropoda"),]

invert_cleanlab = invert_cleanlab[-which(invert_cleanlab$AllTaxa == "Arachnida"),]

invert_cleanlab$Class[which(invert_cleanlab$Subphylum == "Myriapoda")] = "Myriapoda"

invert_cleanlab$Det_Level[which(invert_cleanlab$Class == "Gastropoda")] = "Class"

invert_cleanlab = invert_cleanlab[-which(invert_cleanlab$AllTaxa == "Insecta"),]


AllTaxa = c()
for(i in 1:nrow(invert_cleanlab)){
  if(invert_cleanlab$Det_Level[i] == 'Species'){
    AllTaxa[i] = invert_cleanlab$PON_Name[i]
  }
  else{
    AllTaxa[i] = invert_cleanlab[i,invert_cleanlab$Det_Level[i]]
  }
}
AllTaxa[which(AllTaxa == "indet.")] = "Amblypigi"

invert_cleanlab$AllTaxa = AllTaxa


'%!in%' <- function(x,y)!('%in%'(x,y))

# Fills hierarchy for given single-level labels
longhier = function(zeroclass, hierarchy){
  simpzero = hierarchy[,c('Species',
                          'Genus',
                          'Family',
                          'Order',
                          'Class',
                          'Phylum')]
  
  predont = data.frame(matrix(NA, nrow = length(zeroclass), ncol = 6))
  predont[,1] = zeroclass
  for(i in 2:6){
    for(j in 1:nrow(predont)){
      predont[j,i] = simpzero[which(simpzero[,i-1] == predont[j,i-1])[1], i]
    }
  } 
  colnames(predont) = c("Species",
                        "Genus",
                        "Family",
                        "Order",
                        "Class",
                        "Phylum")
  return(predont)
}

# Fix Amblypigi
invert_cleanlab$AllTaxa[which(invert_cleanlab$AllTaxa == "indet.")] = "Amblypigi"
# Create longlabels df
longlabels = longhier(invert_cleanlab$AllTaxa, hierarchy)

# Determine which taxa are >100 obs
alltable = table(longlabels$Order)
allname = names(which(alltable > 99))

Orders = longlabels$Order
Keep_Orders = Orders[Orders %in% allname]
Keep_Orders = as.factor(Keep_Orders)
Keep_Orders = levels(Keep_Orders)

# Get the taxonomic level a given label is known at (i.e. which level is in Keep_Orders)
getknown = function(simpzero, testlab){
  simpzero$known = NA
  for(i in 1:nrow(simpzero)){
    simpzero$known[i] = which(simpzero[i,] %in% testlab)[1]
  }
  return(simpzero)
}

# Add known column to longlabels
longlabels = getknown(longlabels, Keep_Orders)

invert_cleanlab = invert_cleanlab[-which(is.na(longlabels$known)),]
longlabels = longlabels[-which(is.na(longlabels$known)),]

order_plus = c()
for(i in 1:nrow(longlabels)){
  order_plus[i] = longlabels[i,longlabels$known[i]]
}

invert_cleanlab$order_plus = order_plus

longlab_vector = apply(longlabels[,-c(1:3,7)], 1, function(row) {
  reversed_row <- rev(row)
  paste(reversed_row, collapse = "_")
})
invert_cleanlab$longlab = longlab_vector

################################################################################

# Now for the DNA

#hier_fun from hierarchy.R
dna_hierarchy = hier_fun(edna_rmdup, "LITL", det_level = 'Det_level')

dna_longlabels = longhier(edna_rmdup$LITL, dna_hierarchy)
dna_longlabels$Class[which(dna_longlabels$Class %in% c("Diplopoda", "Chilopoda"))] = "Myriapoda"
dna_longlabels$Order[which(edna_rmdup$Subclass == "Acari")] = "Acari"

# Add known column to longlabels
dna_longlabels = getknown(dna_longlabels, Keep_Orders)


dna_order_plus = c()
for(i in 1:nrow(dna_longlabels)){
  if(is.na(dna_longlabels$known[i])){
    dna_order_plus[i] = NA
  }
  else{
    dna_order_plus[i] = dna_longlabels[i,dna_longlabels$known[i]]
  }
}

edna_rmdup$order_plus = dna_order_plus

dnalonglab_vector = longhier(dna_order_plus, hierarchy)
dnalonglab_vector = apply(dnalonglab_vector[,-c(1:3,7)], 1, function(row) {
  reversed_row <- rev(row)
  paste(reversed_row, collapse = "_")
})
edna_rmdup$longlab = dnalonglab_vector



# #Create long LKTL labels 
# dna_LKTL_Longdf = longhier(ednaLKTL, hierarchy)
# 
# dna_LKTL_Long = apply(dna_LKTL_Longdf[, 1:6], 1, function(row) {
#   paste(rev(row), collapse = "_")
# })
