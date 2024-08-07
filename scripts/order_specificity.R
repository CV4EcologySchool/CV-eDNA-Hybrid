library(stringr)

setwd("C:/Users/jarre/ownCloud/CV-eDNA")

# bool vector of whether or not the predicted class is detected by the DNA 
agreed = c()
# numeric classificaton labels
newclass = read.csv("concat_DNA_named-preds.csv")
newclass = newclass[,1]
numnewclass = read.csv("concat_DNA_preds.csv")
numnewclass = numnewclass[,1]
numnewclass = numnewclass + 1
mhe = read.csv("dna_mhe_order.csv")
valid = read.csv("valid.csv")
train = read.csv("train.csv")

events = mhe$event
mhe = mhe[,-1]

# assigning bool values to agreed
for(x in 1:length(numnewclass)){
  agreed[x] = mhe[which(events == valid$Event[x]),numnewclass[x]]
}
agreed = as.logical(agreed)


################
### DNA Bias ###
################

has_taxa <- function(column) {
  any(column %in% taxa_vector)
}

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

ednaindex = list()
cParent = list()
mrca = c()
splevels = c()
for(x in 1:nrow(valid)){
  if(agreed[x]){
    # Get the row indices in edna_rmdup where the event and LKTL label match the observation
    ednaindex[[x]] = which(edna_rmdup$Event == valid$Event[x] & edna_rmdup$order_plus == newclass[x])
    # Get the taxonomic levels where there is no forking in the edna data (minus NULL or indet. values)
    cParent[[x]] = which(lengths(apply(edna_rmdup[ednaindex[[x]], taxaorder], 2, unique)) == 1
                         & apply(edna_rmdup[ednaindex[[x]], taxaorder], 2, unique) %!in% c("NULL", "indet."))
    # Get the name of the most specific group in CParent that doesn't have forking
    mrca[x] = edna_rmdup[ednaindex[[x]][1], names(cParent[[x]])[1]]
    # Get the specificity of mrca
    splevels[x] = names(cParent[[x]])[1]
    
  }
  else{
    #Get a df of edna data for the current sampling event
    dna_event = edna_rmdup[which(edna_rmdup$Event == valid$Event[x]), taxaorder]
    #Get vector of possible taxa labels (i.e. supertaxa) for current observation
    taxa_vector = unlist(hierarchy[which(hierarchy$Species == newclass[x]),])
    #Logical vector for whether columns in dna_event df have any values in taxa_vector 
    cols = apply(dna_event, 2, has_taxa)
    #If at least one name is detected:
    if(sum(cols) > 0){
      #Get the most specific column
      col = as.numeric(which(cols)[1])
      #Get row indices that contain the name in col
      ednaindex[[x]] = which(dna_event[,col] %in% taxa_vector)
      #Get the name of the ML/DNA overlap taxon
      overlap = unique(dna_event[ednaindex[[x]], col])
      #If overlap is > 1, something is wrong
      if(length(overlap) > 1){
        print("Overlap length > 1")
        
        break
      }
      #This is the same as above (I think)
      cParent[[x]] = which(lengths(apply(dna_event[ednaindex[[x]], taxaorder], 2, unique)) == 1
                           & apply(dna_event[ednaindex[[x]], taxaorder], 2, unique) %!in% c("NULL", "indet."))
      # Get the name of the most specific group in CParent that doesn't have forking
      mrca[x] = dna_event[ednaindex[[x]][1], names(cParent[[x]])[1]]
      # Get the specificity of mrca
      splevels[x] = names(cParent[[x]])[1]
      
    }
    #If no names are detected (i.e. the DNA did not detect it even at phylum level), leave classification unchanged
    else{
      ednaindex[[x]] = NA
      cParent[[x]] = NA
      mrca[x] = newclass[x]
      splevels[x] = NA
    }
  }
}

# This could cause some problems if Det_Level hasn't been adjusted
# for order level labels. But in this case it seems fine.
# Bigger issue actually is that I'm using valid instead of newclass,
# but again, it is fine in this case.
splevels[is.na(splevels)] = valid[is.na(splevels),"Det_Level"]

# #The following does the analysis only for cases where the DNA and ML agree
# 
# splevels_agreed = splevels[agreed]
# mrca_agreed = mrca[agreed]
# splevels_agreed = str_to_title(splevels_agreed)
# 
# newclass_agreed = newclass[agreed]
# 
# taxa = invertmatch[,28:40]
# 
# last_col_index = c()
# for (i in 1:nrow(train)){
#   last_col_index[i] <- max(which(taxa[i,] == train$order_plus[i]))
# }
# Det_Level = colnames(taxa)[last_col_index]
# 
# splevelsog_agreed = c()
# for(x in 1:length(newclass_agreed)){
#   splevelsog_agreed[x] = as.character(Det_Level[which(train$order_plus == newclass_agreed[x])[1]])
# }
# Det_Level = as.factor(Det_Level)
# splevelsog_agreed = factor(splevelsog_agreed, levels = levels(Det_Level))
# #splevelsog_agreed = splevelsog_agreed[agreed]
# 
# numsplevels_agreed = as.numeric(ordered(splevels_agreed, levels = taxaorder))
# numsplevelsog_agreed = as.numeric(ordered(splevelsog_agreed, levels = taxaorder))
# 
# SpecifyImprove_agreed = data.frame(numsplevelsog_agreed, numsplevels_agreed)
# improve_agreed = SpecifyImprove_agreed[,1] - SpecifyImprove_agreed[,2]

###########
### All ###
###########

#The following does the entire DNA bias analysis

splevels = str_to_title(splevels)

taxa = invertmatch[invertmatch$X %in% train$X,28:40]
taxa[taxa$Class == "Collembola", "Order"] = "Collembola"

last_col_index = c()
for (i in 1:nrow(train)){
  last_col_index[i] <- max(which(taxa[i,] == train$order_plus[i]))
}
Det_Level = colnames(taxa)[last_col_index]

splevelsog = c()
for(x in 1:length(newclass)){
  splevelsog[x] = as.character(Det_Level[which(train$order_plus == newclass[x])[1]])
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
### Just Agreed  ###
####################

splevels_agreed = splevels[agreed]
splevelsog_agreed = splevelsog[agreed]

numsplevels_agreed = as.numeric(ordered(splevels_agreed, levels = taxaorder))
numsplevelsog_agreed = as.numeric(ordered(splevelsog_agreed, levels = taxaorder))

SpecifyImprove_agreed = data.frame(numsplevelsog_agreed, numsplevels_agreed)
improve_agreed = SpecifyImprove_agreed[,1] - SpecifyImprove_agreed[,2]

####################
### ML Bias Full ###
####################

mrca_mlbias = mrca
mrca_mlbias[!agreed] = newclass[!agreed]

splevels_mlbias = splevels
splevels_mlbias[!agreed] = as.character(splevelsog[!agreed])

numsplevels_mlbias = as.numeric(ordered(splevels_mlbias, levels = taxaorder))
numsplevelsog = as.numeric(ordered(splevelsog, levels = taxaorder))

SpecifyImprove_mlbias = data.frame(numsplevelsog, numsplevels_mlbias)
improve_mlbias = SpecifyImprove_mlbias[,1] - SpecifyImprove_mlbias[,2]

###################################################################################################

SpecifyImprove$mrca = mrca
unique_taxa <- SpecifyImprove %>%
  count(numsplevels, mrca)

SpecifyImprove_mlbias$mrca = mrca_mlbias
unique_taxa_mlbias <- SpecifyImprove_mlbias %>%
  count(numsplevels_mlbias, mrca)

unique_taxa_merged = merge(unique_taxa, unique_taxa_mlbias, by = "mrca", all = T)

write.csv(unique_taxa, "unique_taxa_dnabias.csv", row.names = F)
write.csv(unique_taxa_mlbias, "unique_taxa_mlbias.csv", row.names = F)
write.csv(unique_taxa_merged, "unique_taxa_merged.csv", row.names = F)

###################################################################################################

og_prop = table(factor(SpecifyImprove[,1], levels = c(1:13)))/nrow(SpecifyImprove)
og_prop_agreed = table(factor(SpecifyImprove_agreed[,1], levels = c(1:13)))/nrow(SpecifyImprove_agreed)

improve_prop = table(factor(SpecifyImprove[,2], levels = c(1:13)))/nrow(SpecifyImprove)
improve_prop_agreed = table(factor(SpecifyImprove_agreed[,2], levels = c(1:13)))/nrow(SpecifyImprove_agreed)
improve_prop_mlbias = table(factor(SpecifyImprove_mlbias[,2], levels = c(1:13)))/nrow(SpecifyImprove_mlbias)

sum(og_prop[1:8])
sum(og_prop_agreed[1:8])
sum(improve_prop[1:8])
sum(improve_prop_agreed[1:8])
sum(improve_prop_mlbias[1:8])

sum(og_prop[11:13])
sum(og_prop_agreed[11:13])
sum(improve_prop[11:13])
sum(improve_prop_agreed[11:13])
sum(improve_prop_mlbias[11:13])

# Mike_og_prop = table(factor(SpecifyImprove[,1], levels = c(1:13)))/nrow(SpecifyImprove)
# Mike_og_prop_agreed = table(factor(SpecifyImprove_agreed[,1], levels = c(1:13)))/nrow(SpecifyImprove_agreed)
# 
# Mike_improve_prop = table(factor(SpecifyImprove[,2], levels = c(1:13)))/nrow(SpecifyImprove)
# Mike_improve_prop_agreed = table(factor(SpecifyImprove_agreed[,2], levels = c(1:13)))/nrow(SpecifyImprove_agreed)
# Mike_improve_prop_mlbias = table(factor(SpecifyImprove_mlbias[,2], levels = c(1:13)))/nrow(SpecifyImprove_mlbias)

###################################################################################################

# specify_cols = c("Original", "New")
# 
# SpecifyImprove = as.data.frame(matrix(taxaorder[as.matrix(SpecifyImprove)], ncol = 2))
# SpecifyImprove_agreed = as.data.frame(matrix(taxaorder[as.matrix(SpecifyImprove_agreed)], ncol = 2))
# SpecifyImprove_mlbias = as.data.frame(matrix(taxaorder[as.matrix(SpecifyImprove_mlbias)], ncol = 2))
# 
# 
# colnames(SpecifyImprove) = specify_cols
# colnames(SpecifyImprove_agreed) = specify_cols
# colnames(SpecifyImprove_mlbias) = specify_cols
# 
# 
# write.csv(SpecifyImprove, "Mike_specify_DNA_bias.csv", row.names = F)
# write.csv(SpecifyImprove_agreed, "Mike_specify_agreed.csv", row.names = F)
# write.csv(SpecifyImprove_mlbias, "Mike_specify_ML_bias.csv", row.names = F)



