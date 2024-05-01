# Initializes invert_cleanlab dataframe. This removes some junk groups like 'Arthropoda'
# It also removes underrepresented Orders while merging some orders with their respective Class
# or Phylum labels (e.g. Annelida)

setwd("C:/Users/jarre/ownCloud/CV-eDNA")

invertmatch = read.csv("invertmatch.csv")
hierarchy = read.csv("hierarchy_order.csv")


invert_cleanlab = invertmatch[-which(invertmatch$AllTaxa == "Arthropoda"),]

invert_cleanlab = invert_cleanlab[-which(invert_cleanlab$AllTaxa == "Arachnida"),]

invert_cleanlab$Class[which(invert_cleanlab$Subphylum == "Myriapoda")] = "Myriapoda"

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



# #Create long LKTL labels 
# dna_LKTL_Longdf = longhier(ednaLKTL, hierarchy)
# 
# dna_LKTL_Long = apply(dna_LKTL_Longdf[, 1:6], 1, function(row) {
#   paste(rev(row), collapse = "_")
# })
