library(stringr)

setwd("C:/Carabid_Data/CV-eDNA/splits/LKTL-37141")

# bool vector of whether or not the predicted class is detected by the DNA 
agreed = c()
# numeric classificaton labels
newclass = read.csv("preds_concat_named.csv")
newclass = newclass[,1]
numnewclass = read.csv("preds_concat.csv")
numnewclass = numnewclass[,1]
numnewclass = numnewclass + 1
mhe = read.csv("naive.csv")
valid = read.csv("valid.csv")
train = read.csv("train.csv")

events = mhe$edna
mhe = mhe[,-1]

# assigning bool values to agreed
for(x in 1:length(numnewclass)){
  agreed[x] = mhe[which(events == valid$Event[x]),numnewclass[x]]
}
agreed = as.logical(agreed)


#cleaning names 
colnames(ednamatch)[which(colnames(ednamatch) == "oorder")] = "order"
colnames(ednamatch) = str_to_title(colnames(ednamatch))
ednamatch$Det_level = str_to_title(ednamatch$Det_level)

ednamatch$Percent_sim = as.numeric(ednamatch$Percent_sim)
ednamatch$Percent_sim[which(ednamatch$Percent_sim == 0)] = 100
ednamatch = ednamatch[which(ednamatch$Percent_sim >= 97),]

ednamatch$Species[which(grepl("indet", ednamatch$Speciesname))] = "indet."
ednamatch$Det_level[which(grepl("indet", ednamatch$Speciesname))[c(1:7,10,11)]] = "Genus"
ednamatch$Det_level[which(grepl("indet", ednamatch$Speciesname))[c(8,9)]] = "Family"
ednamatch$Speciesname[which(grepl("indet", ednamatch$Speciesname))] = NA
ednamatch$Species[ednamatch$Det_level == "Species"] = ednamatch$Speciesname[ednamatch$Det_level == "Species"]
ednaLITL = ednamatch[cbind(1:nrow(ednamatch), match(ednamatch$Det_level, colnames(ednamatch)))]
ednamatch$Det_level[which(ednaLITL == "indet.")] = "Family"
ednaLITL = ednamatch[cbind(1:nrow(ednamatch), match(ednamatch$Det_level, colnames(ednamatch)))]
ednamatch$LITL = ednaLITL

ednamatch = ednamatch[-which(ednamatch$Det_level == "Phylum"),]

ednamatch$Superfamily[ednamatch$Superfamily == "Blaberoidea"] = "Blattoidea"
ednamatch$Family[ednamatch$Genus == "Parcoblatta"] = "Ectobiidae"
ednamatch$Subfamily[ednamatch$Genus == "Parcoblatta"] = "Blattellinae"

# longhier function comes from LKTL_Scheme.R
LITL_Longdf = longhier(ednamatch$LITL, hierarchy)
ednamatch$Event[which(ednamatch$Tubes == "MOAB00420161004")] = "MOAB00420161004"
LITL_Longdf$LITL = ednamatch$LITL
LITL_Longdf$Event = ednamatch$Event

LKTL_Long = apply(LITL_Longdf[, 7:8], 1, function(row) {
  paste(rev(row), collapse = "_")
})

edna_rmdup = ednamatch[-which(duplicated(LKTL_Long)),]

rm_row = c()
for(i in which(edna_rmdup$Det_level != "Species")){
  Det_level = edna_rmdup$Det_level[i]
  i_event = edna_rmdup$Event[i]
  taxon = edna_rmdup[cbind(i, match(Det_level, colnames(edna_rmdup)))]
  if(sum(edna_rmdup[edna_rmdup$Event == i_event & edna_rmdup[,Det_level] == taxon, "Det_level"] == "Species") > 0){
    rm_row = c(rm_row, i)
  }
}

edna_rmdup = edna_rmdup[-rm_row,]

hierarchylevels = c("Subfamily", 
                    "Family", 
                    "Superfamily", 
                    "Order", 
                    "Class")

rm_row = c()
for(h_level in hierarchylevels){
  for(i in which(edna_rmdup$Det_level == h_level)){
    Det_level = edna_rmdup$Det_level[i]
    i_event = edna_rmdup$Event[i]
    taxon = edna_rmdup[cbind(i, match(Det_level, colnames(edna_rmdup)))]
    if(sum(edna_rmdup[edna_rmdup$Event == i_event, Det_level] == taxon) > 1){
      rm_row = c(rm_row, i)
    }
  }
}

edna_rmdup = edna_rmdup[-rm_row,]

edna_rmdup$LKTL = ednaLITL


'%!in%' <- function(x,y)!('%in%'(x,y))

hierarchylevels = c("Phylum", 
                    "Subphylum", 
                    "Class", 
                    "Subclass", 
                    "Superorder", 
                    "Order", 
                    "Suborder", 
                    "Infraorder", 
                    "Superfamily", 
                    "Family", 
                    "Subfamily", 
                    "Genus", 
                    "Species")


ednaindex = list()
cParent = list()
mrca = c()
splevels = c()
for(x in 1:nrow(valid)){
  if(agreed[x]){
    # Get the row indices in edna_rmdup where the event and LKTL label match the observation
    ednaindex[[x]] = which(edna_rmdup$Event == valid$Event[x] & edna_rmdup$LKTL == newclass[x])
    # Get the taxonomic levels where there is no forking in the edna data (minus NULL or indet. values)
    cParent[[x]] = which(lengths(apply(edna_rmdup[ednaindex[[x]], hierarchylevels], 2, unique)) == 1
                           & apply(edna_rmdup[ednaindex[[x]], hierarchylevels], 2, unique) %!in% c("NULL", "indet."))
    # Get the name of the most specific group in CParent that doesn't have forking
    mrca[x] = edna_rmdup[ednaindex[[x]][1], names(cParent[[x]])[length(cParent[[x]])]]
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
    # Get the row indices in edna_rmdup where the event and LKTL label match the observation
    ednaindex[[x]] = which(edna_rmdup$Event == valid$Event[x] & edna_rmdup$LKTL == newclass[x])
    # Get the taxonomic levels where there is no forking in the edna data (minus NULL or indet. values)
    cParent[[x]] = which(lengths(apply(edna_rmdup[ednaindex[[x]], hierarchylevels], 2, unique)) == 1
                         & apply(edna_rmdup[ednaindex[[x]], hierarchylevels], 2, unique) %!in% c("NULL", "indet."))
    # Get the name of the most specific group in CParent that doesn't have forking
    mrca[x] = edna_rmdup[ednaindex[[x]][1], names(cParent[[x]])[length(cParent[[x]])]]
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











  
  