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
  return(predont)
}

# Fix Amblypigi
invertmatch$AllTaxa[which(invertmatch$AllTaxa == "indet.")] = "Amblypigi"
# Create longlabels df
longlabels = longhier(invertmatch$AllTaxa, hierarchy)

# Determine which taxa are >100 obs
alltable = table(invertmatch$AllTaxa)
allname = names(which(alltable <= 99))

AllTaxa = invertmatch$AllTaxa
KeepTaxa = AllTaxa[AllTaxa %!in% allname]
KeepTaxa = as.factor(KeepTaxa)

# Get the taxonomic level a given label is known at (i.e. which level is in KeepTaxa)
getknown = function(simpzero, testlab){
  simpzero$known = NA
  for(i in 1:nrow(simpzero)){
    simpzero$known[i] = which(simpzero[i,] %in% levels(testlab))[1]
  }
  return(simpzero)
}

# Add known column to longlabels
longlabels = getknown(longlabels, KeepTaxa)

# Create LKTL labels for all specimens
LKTL = c()
for(i in 1:nrow(longlabels)){
  LKTL[i] = longlabels[i,longlabels$known[i]]
}

#Visualize LKTL labels
LKTL_Table = as.data.frame(table(LKTL))
LKTL_Table = LKTL_Table[order(LKTL_Table$Freq, decreasing = TRUE), ]

barplot(LKTL_Table$Freq, names.arg = LKTL_Table$LKTL, 
        main = "Abundance of Categories",
        xlab = "Categories", ylab = "Abundance",
        col = "blue")


#Create long LKTL labels 
LKTL_Longdf = longhier(LKTL, hierarchy)

LKTL_Long = apply(LKTL_Longdf[, 1:6], 1, function(row) {
  paste(rev(row), collapse = "_")
})



