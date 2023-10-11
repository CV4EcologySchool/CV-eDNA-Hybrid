

sprow = which(ednaclean$det_level == "species")
ednaclean[sprow, "species"] = paste(ednaclean$genus[sprow], ednaclean$species[sprow])

'%!in%' <- function(x,y)!('%in%'(x,y))

LKTL_Long = levels(as.factor(LKTL_Long))
LKTL_ordered = sapply(strsplit(LKTL_Long, "_"), function(x) x[length(x)])

rem = c("indet.", "NULL")
dnabias_list = list()

for(event in event_names){
  dna_event = unique(as.vector(as.matrix(ednaclean[which(ednaclean$event == event), 1:13])))
  dna_event = dna_event[which(dna_event %!in% rem)]
  x = LKTL_ordered %in% dna_event
  dnabias_list[[event]] = as.integer(x)
}

dnabias_mhe = data.frame(do.call(rbind, dnabias_list))

has_taxa <- function(column) {
  any(column %in% taxa_vector)
}

pb = progress_bar$new(
  format = "[:bar] :percent Elapsed: :elapsed Time remaining: :eta",
  total = nrow(valid)
)

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
    dna_event = ednaclean[which(ednaclean$event == valid$Event[x]), 1:13]
    taxa_vector = unlist(hierarchy[which(hierarchy$Species == newclass[x]),])
    cols = apply(dna_event, 2, has_taxa)
    if(sum(cols) > 0){
      col = as.numeric(tail(which(cols), n =1))
      ednaindex[[x]] = which(dna_event[,col] %in% taxa_vector)
      overlap = unique(dna_event[ednaindex[[x]], col])
      if(length(overlap) > 1){
        print("Overlap length > 1")
        
        break
      }
      cParent[[x]] = which(lengths(apply(dna_event[ednaindex[[x]], c(1:13)], 2, unique)) == 1
                           & apply(dna_event[ednaindex[[x]], c(1:13)], 2, unique) %!in% c("NULL", "indet."))
      # Get the name of the most specific group in CParent that doesn't have forking
      mrca[x] = dna_event[ednaindex[[x]][1], names(cParent[[x]])[length(cParent[[x]])]]
      # Get the specificity of mrca
      splevels[x] = names(cParent[[x]])[length(cParent[[x]])]
      
    }
    else{
      ednaindex[[x]] = NA
      cParent[[x]] = NA
      mrca[x] = newclass[x]
      splevels[x] = valid$Det_Level[x]
    }
  }
}

splevels_agreed = splevels[agreed]
mrca_agreed = mrca[agreed]
splevels_agreed = str_to_title(splevels_agreed)
splevels_agreed[which(mrca_agreed == "Myriapoda")] = "Class"

newclass_agreed = newclass[agreed]

train_taxa = train[,28:40]
train_taxa$Species[which(train$LKTL == "Canthon viridis")] = "Canthon viridis"

last_col_index = c()
for (i in 1:nrow(train)){
  last_col_index[i] <- max(which(train_taxa[i,] == train$LKTL[i]))
}
Det_Level = colnames(train_taxa)[last_col_index]

splevelsog_agreed = c()
for(x in 1:length(newclass_agreed)){
  splevelsog_agreed[x] = as.character(Det_Level[which(train$LKTL == newclass_agreed[x])[1]])
}
Det_Level = as.factor(Det_Level)
splevelsog_agreed = factor(splevelsog_agreed, levels = levels(Det_Level))
#splevelsog_agreed = splevelsog_agreed[agreed]


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

numsplevels_agreed = as.numeric(ordered(splevels_agreed, levels = taxaorder))
numsplevelsog_agreed = as.numeric(ordered(splevelsog_agreed, levels = taxaorder))

SpecifyImprove_agreed = data.frame(numsplevelsog_agreed, numsplevels_agreed)
improve_agreed = SpecifyImprove[,1] - SpecifyImprove[,2]

###########
### All ###
###########


splevels = str_to_title(splevels)
splevels[which(mrca == "Myriapoda")] = "Class"

train_taxa = train[,28:40]
train_taxa$Species[which(train$LKTL == "Canthon viridis")] = "Canthon viridis"

last_col_index = c()
for (i in 1:nrow(train)){
  last_col_index[i] <- max(which(train_taxa[i,] == train$LKTL[i]))
}
Det_Level = colnames(train_taxa)[last_col_index]

splevelsog = c()
for(x in 1:length(newclass)){
  splevelsog[x] = as.character(Det_Level[which(train$LKTL == newclass[x])[1]])
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


####################
### ML Bias Full ###
####################

splevels_mlbias = splevels
splevels_mlbias[!agreed] = as.character(splevelsog[!agreed])

numsplevels_mlbias = as.numeric(ordered(splevels_mlbias, levels = taxaorder))
numsplevelsog = as.numeric(ordered(splevelsog, levels = taxaorder))

SpecifyImprove_mlbias = data.frame(numsplevelsog, numsplevels_mlbias)
improve_mlbias = SpecifyImprove_mlbias[,1] - SpecifyImprove_mlbias[,2]


