#This code initializes the invertmatch and ednamatch dataframes

library(stringr)

#Match invert$Event format to edna$event format
setwd("C:/Users/jarre/ownCloud/CV-eDNA")

invert = read.csv("TaxMorColWea4.1.csv")
edna = read.csv("eDNARaw.csv")

cleanevents = gsub('_', '', invert$Event)
cleanevents = gsub('\\.', '', cleanevents)
invert$Event = cleanevents

#Find which events overlap between invert & edna and which don't
mlevents = levels(as.factor(invert$Event))
ednaevents = levels(as.factor(edna$event))
edna_in_events = ednaevents %in% mlevents
ml_in_events = mlevents %in% ednaevents
missing_edna_events = ednaevents[which(!edna_in_events)]
missing_ml_events = mlevents[which(!ml_in_events)]


#Remove non-matching events
ednamatch = edna[-which(edna$event %in% missing_edna_events),]
invertmatch = invert[-which(invert$Event %in% missing_ml_events),]

#Remove low quality DNA detections
ednamatch$percent_sim = as.numeric(ednamatch$percent_sim)
ednamatch = ednamatch[-which(ednamatch$percent_sim > 0 & ednamatch$percent_sim < 97),]
ednamatch = ednamatch[-which(ednamatch$det_level == "phylum"),]

#Clean invert dataset by removing ignore and juveniles
invertmatch = invertmatch[-which(grepl("\\bIgnore\\b", invertmatch$PON_Name, ignore.case = TRUE)),]
invertmatch = invertmatch[-which(grepl("\\bNo Clue\\b", invertmatch$PON_Name, ignore.case = TRUE)),]
invertmatch = invertmatch[-which(grepl("\\bLarva(e)?\\b", invertmatch$PON_Name, ignore.case = TRUE)),]
invertmatch = invertmatch[-which(grepl("\\bjuvenile\\b", invertmatch$PON_Name, ignore.case = TRUE)),]
invertmatch = invertmatch[-which(grepl("\\bNymph\\b(?![a-z])", invertmatch$PON_Name, ignore.case = TRUE, perl = TRUE)),]

#Continue cleaning by only keeping intact/complete observations
keep = c('complete',
         'Complete',
         'Fly',
         'intact')
invertmatch = invertmatch[invertmatch$damaged %in% keep,]
invertmatch = invertmatch[-which(invertmatch$detBy == "KMS"),]

invertmatch$Det_Level = str_to_title(invertmatch$Det_Level)

#Create AllTaxa
AllTaxa = c()
for(i in 1:nrow(invertmatch)){
  if(invertmatch$Det_Level[i] == 'Species'){
    AllTaxa[i] = invertmatch$PON_Name[i]
  }
  else{
    AllTaxa[i] = invertmatch[i,invertmatch$Det_Level[i]]
  }
}


invertmatch$PON_Name[AllTaxa == "indet."]
#Only do the following two lines of code if the prior shows "Amblypigi indet. suborder" twice
invertmatch$Det_Level[AllTaxa == "indet."] = "Order"
AllTaxa[which(AllTaxa == "indet.")] = "Amblypigi"

#Make a SpeciesName column for ednamatch
ednamatch$SpeciesName = NA
ednamatch$SpeciesName[ednamatch$det_level == "species"] = ednamatch$PON_Name[ednamatch$det_level == "species"]

invertmatch$AllTaxa = AllTaxa

entiminae_idx = which(invertmatch$Subfamily == "Entiminae")
invertmatch$Det_Level[entiminae_idx] = "Subfamily"
invertmatch$AllTaxa[entiminae_idx] = "Entiminae"


write.csv(invertmatch, "invertmatch.csv")
write.csv(ednamatch, "ednamatch.csv")
