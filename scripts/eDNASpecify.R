library(stringr)

# bool vector of whether or not the predicted class is detected by the DNA 
agreed = c()
# numeric classificaton labels
newclass = read.csv("preds_concat_named.csv")
newclass = newclass[,1]
numnewclass = read.csv("preds_concat.csv")
numnewclass = numnewclass[,1]
numnewclass = numnewclass + 1
mhe = read.csv("./splits/LKTL-37141/naive.csv")
valid = read.csv("./splits/LKTL-37141/valid.csv")
train = read.csv("./splits/LKTL-37141/train.csv")

events = mhe$event
mhe = mhe[,-1]

# assigning bool values to agreed
for(x in 1:length(numnewclass)){
  agreed[x] = mhe[which(events == valid$Event[x]),numnewclass[x]]
}
agreed = as.logical(agreed)

#cleaning names 
ednaclean$det_level[which(ednaclean$det_level == "Subfamily")] = "subfamily"
colnames(ednaclean)[which(colnames(ednaclean) == "oorder")] = "order"


'%!in%' <- function(x,y)!('%in%'(x,y))

ednaindex = list()
cParent = list()
mrca = c()
splevels = c()
for(x in 1:nrow(valid)){
  if(agreed[x]){
    # Get the row indices in ednaclean where the event and LKTL label match the observation
    ednaindex[[x]] = which(ednaclean$event == valid$Event[x] & ednaclean$LKTL == newclass[x])
    # Get the taxonomic levels where there is no forking in the edna data (minus NULL or indet. values)
    cParent[[x]] = which(lengths(apply(ednaclean[ednaindex[[x]], c(1:13)], 2, unique)) == 1
                           & apply(ednaclean[ednaindex[[x]], c(1:13)], 2, unique) %!in% c("NULL", "indet."))
    # Get the name of the most specific group in CParent that doesn't have forking
    mrca[x] = ednaclean[ednaindex[[x]][1], names(cParent[[x]])[length(cParent[[x]])]]
    # Get the specificity of mrca
    splevels[x] = names(cParent[[x]])[length(cParent[[x]])]
  }
  else{
    ednaindex[[x]] = NA
    cParent[[x]] = NA
    mrca[x] = NA
  }
}
splevels = splevels[agreed]
mrca_agreed = mrca[agreed]
splevels = str_to_title(splevels)
splevels[which(mrca_agreed == "Myriapoda")] = "Class"


newclass_agreed = newclass[agreed]

train_taxa = train[,28:40]
train_taxa$Species[which(train$LKTL == "Canthon viridis")] = "Canthon viridis"

last_col_index = c()
for (i in 1:nrow(train)){
  last_col_index[i] <- max(which(train_taxa[i,] == train$LKTL[i]))
}
Det_Level = colnames(train_taxa)[last_col_index]

splevelsog = c()
for(x in 1:length(newclass_agreed)){
  splevelsog[x] = as.character(Det_Level[which(train$LKTL == newclass_agreed[x])[1]])
}
Det_Level = as.factor(Det_Level)
splevelsog = factor(splevelsog, levels = levels(Det_Level))
#splevelsog = splevelsog[agreed]


taxaorder = c("Species",
              "Genus",
              "Subfamily",
              "Family",
              "Superfamily",
              "Infraorder",
              "Suborder",
              "Order",
              "Superorder",
              "Subclass",
              "Class",
              "Subphylum",
              "Phylum")
 
numsplevels = as.numeric(ordered(splevels, levels = taxaorder))
numsplevelsog = as.numeric(ordered(splevelsog, levels = taxaorder))

SpecifyImprove = data.frame(numsplevelsog, numsplevels)
improve = SpecifyImprove[,1] - SpecifyImprove[,2]

##################
#### DNA Bias ####
##################

hier_newclass = as.data.frame(matrix(NA, nrow = length(newclass), ncol = ncol(hierarchy)))

for(i in 1:length(newclass)){
  hier_newclass[i,] = hierarchy[which(hierarchy$Species == newclass[i]),]
}


ednaindex = list()
cParent = list()
mrca = c()
splevels = c()
for(x in 1:nrow(valid)){
  if(agreed[x]){
    # Get the row indices in ednaclean where the event and LKTL label match the observation
    ednaindex[[x]] = which(ednaclean$event == valid$Event[x] & ednaclean$LKTL == newclass[x])
    # Get the taxonomic levels where there is no forking in the edna data (minus NULL or indet. values)
    cParent[[x]] = which(lengths(apply(ednaclean[ednaindex[[x]], c(1:13)], 2, unique)) == 1
                         & apply(ednaclean[ednaindex[[x]], c(1:13)], 2, unique) %!in% c("NULL", "indet."))
    # Get the name of the most specific group in CParent that doesn't have forking
    mrca[x] = ednaclean[ednaindex[[x]][1], names(cParent[[x]])[length(cParent[[x]])]]
    # Get the specificity of mrca
    splevels[x] = names(cParent[[x]])[length(cParent[[x]])]
  }
  else{
    ednaindex[[x]] = NA
    cParent[[x]] = NA
    mrca[x] = NA
  }
}
splevels = splevels[agreed]
splevels = str_to_title(splevels)











  
  