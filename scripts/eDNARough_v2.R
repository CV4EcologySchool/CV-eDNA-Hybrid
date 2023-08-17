library(stringr)

#Match invert$Event format to edna$event format
setwd("C:/Users/jarre/ownCloud/CV-eDNA")

invert = read.csv("TaxMorColWea4.1.csv")
invertmatch = read.csv("invertmatch.csv")
edna = read.csv("eDNARaw.csv")

test = gsub('_', '', invert$Event)
test = gsub('\\.', '', test)
invert$Event = test

#Find which events overlap between invert & edna and which don't
mllevels = levels(as.factor(invert$Event))
ednalevels = levels(as.factor(edna$event))
ednain = ednalevels %in% mllevels
mlin = mllevels %in% ednalevels
missingedna = ednalevels[which(!ednain)]
missingml = mllevels[which(!mlin)]

# missingml[56] = NA
# test = data.frame(missingedna, missingml)

#Remove non-matching events
ednamatch = edna[-which(edna$event %in% missingedna),]
invertmatch = invert[-which(invert$Event %in% missingml),]


mlinlevels = mllevels[which(mlin)]

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

# invertmatch = invertmatch[-which(invertmatch$AllTaxa %in% c("Ignore",
#                                                "Termite",
#                                                "indet.",
#                                                "Juvenile",
#                                                "Larva",
#                                                "Nymph",
#                                                NA)),]

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

#This loop sees if AllTaxa is present in the eDNA list for each sampling event
ednamatch = as.matrix(ednamatch)
match = c()
i = 1
phold = invertmatch[i,"Event"]
subset_edna = ednamatch[which(ednamatch[,"event"] == invertmatch[i,"Event"]), c(13:25, 41)]
for(i in 1:nrow(invertmatch)){
  #This if statement makes the whole thing go way faster
  if(phold != invertmatch[i,"Event"]){
    phold = invertmatch[i,"Event"]
    subset_edna = ednamatch[which(ednamatch[,"event"] == invertmatch[i,"Event"]), c(13:25, 41)]
  }
  match[i] = AllTaxa[i] %in% subset_edna
}

#Calculates eDNA TP rate (even though the name is FP). Doesn't seem to be used though
invertmatch$AllTaxa = AllTaxa
ednafp = c()
for(i in 1:nrow(ednamatch)){
  ednafp[i] = any(ednamatch[i, c(13:25, 41)] %in% invertmatch[which(invertmatch[,"Event"] == ednamatch[i,"event"]), "AllTaxa"])
}

##################


ednaclean = ednamatch[,c(13:26,28,29)]
ednaclean = ednaclean[!duplicated(ednaclean[c(1:nrow(ednaclean)),]),] #For some reason I have to specify the rows or else it doesn't work

test = ednamatch[!duplicated(ednamatch[c(1:nrow(ednamatch)),]),]

LITLs = unique(invertmatch$AllTaxa)
ednaLITL = c()
eventmatch = c()
for(i in 1:nrow(ednaclean)){
  ednaLITL[i] = ednaclean[i,max(which(ednaclean[i,] %in% LITLs))]
  eventmatch[i] = ednaLITL[i] %in% invertmatch[which(invertmatch[,"Event"] == ednaclean[i,"event"]), "AllTaxa"]
}

ednaLITL = factor(ednaLITL, levels = LITLs)
ednaclean = data.frame(ednaclean, ednaLITL)

eventclean = ednaclean[!duplicated(ednaclean[c(1:nrow(ednaclean)),c(16,17)]),]
                      
#known is AllTaxa levels from train data

ednaknown = c()
eventmatch = c()
for(i in 1:nrow(ednaclean)){
  ednaknown[i] = ednaclean[i,max(which(ednaclean[i,] %in% known))]
}

ednaknown = factor(ednaknown, levels = known)
ednaclean = data.frame(ednaclean, ednaknown)
eventclean = ednaclean[!duplicated(ednaclean[c(1:nrow(ednaclean)),c(16,18)]),]


imageknown = c()
for(i in 1:nrow(invertmatch)){
  imageknown[i] = invertmatch[i,max(which(invertmatch[i,] %in% known))]
}

invertmatch = data.frame(invertmatch, imageknown)
eventimage = invertmatch[!duplicated(invertmatch[c(1:nrow(invertmatch)),c("Event","imageknown")]),]

eventmatch = c()
for(i in 1:nrow(eventclean)){
  eventmatch[i] = eventclean$ednaknown[i] %in% eventimage[which(eventimage[,"Event"] == eventclean[i,"event"]), "imageknown"]
}

# eventimage = list()
# for(event in mlinlevels){
#   eventimage[[event]] = unique(invertmatch$AllTaxa[which(invertmatch$Event == event)])
# }
# eventimage = stack(eventimage)
# colnames(eventimage) = c("imageknown", "Event")

imagetp = c()
for(i in 1:nrow(eventimage)){
  imagetp[i] = eventimage$imageknown[i] %in% eventclean[which(eventclean[,"event"] == eventimage[i,"Event"]), "ednaknown"]
}


imageknowntab = table(eventimage$imageknown)

imageoverlaptab = table(eventimage$imageknown[imagetp])

test = data.frame(LITLs,
                  imageknowntab, 
                  as.numeric(imageoverlaptab), 
                  (as.numeric(imageknowntab) - as.numeric(imageoverlaptab)),
                  (as.numeric(imageoverlaptab) / as.numeric(imageknowntab)),
                  as.numeric(ednaknowntab),
                  as.numeric(ednaoverlaptab),
                  (as.numeric(ednaknowntab) - as.numeric(ednaoverlaptab)),
                  (as.numeric(ednaoverlaptab) / as.numeric(ednaknowntab)))

colnames(test) = c("LITL",
                   "InImage",
                   "ImageTP",
                   "ImageFN",
                   "Recall",
                   "InEDNA",
                   "eDNATP",
                   "eDNAFN",
                   "Precision")

