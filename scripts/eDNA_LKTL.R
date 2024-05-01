
setwd("C:/Users/jarre/ownCloud/CV-eDNA")

ednamatch = read.csv("ednamatch.csv")

ednaclean = ednamatch[,c(14:27,29,30,42)]
# Code below used to be ednaclean = ednaclean[...]. Change back for stuff down the line possibly
edna_rmdup = ednamatch[!duplicated(ednaclean[c(1:nrow(ednaclean)),]),] #For some reason I have to specify the rows or else it doesn't work

LKTLs = unique(invert_cleanlab$LKTL)
ednaLKTL = c()
eventmatch = c()
for(i in 1:nrow(edna_rmdup)){
  ednaLKTL[i] = edna_rmdup[i,max(which(edna_rmdup[i,] %in% LKTLs))]
  eventmatch[i] = ednaLKTL[i] %in% invert_cleanlab[which(invert_cleanlab[,"Event"] == edna_rmdup[i,"event"]), "LKTL"]
}

ednaLKTL = factor(ednaLKTL, levels = LKTLs)
ohe = data.frame(model.matrix(~ ednaLKTL - 1))
colnames(ohe) = sub("ednaLKTL", "", colnames(ohe))
ohe$event = ednamatch$event

ohe_rmdup = ohe[!duplicated(ednaclean[c(1:nrow(ednaclean)),]),]

mhe <- ohe %>%
  group_by(event) %>%
  summarise_all(sum)

mhe_rmdup = ohe_rmdup %>%
  group_by(event) %>%
  summarise_all(sum)

write.csv(mhe, "mhe.csv", row.names = F)
write.csv(mhe_rmdup, "mhe_lite.csv", row.names = F)


# This next chunk of code makes a plot showing the distribution of how many hits
# a group gets when it is detected (i.e. how many forks if it is present). I need
# to modify this to include nesting (e.g. Arthropoda shows an average of 1 hit,
# which obviously isn't true)
library(reshape2)
molten_mhe = melt(mhe_rmdup, id.vars = "event", variable.name = "Category", value.name = "Value")
molten_mhe <- molten_mhe %>%
  filter(Value > 0)

ggplot(molten_mhe, aes(x = Category, y = Value)) +
  geom_boxplot() +
  labs(x = "Category", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


ednaclean = data.frame(ednaclean, ednaLKTL)

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

test = data.frame(LKTLs,
                  imageknowntab, 
                  as.numeric(imageoverlaptab), 
                  (as.numeric(imageknowntab) - as.numeric(imageoverlaptab)),
                  (as.numeric(imageoverlaptab) / as.numeric(imageknowntab)),
                  as.numeric(ednaknowntab),
                  as.numeric(ednaoverlaptab),
                  (as.numeric(ednaknowntab) - as.numeric(ednaoverlaptab)),
                  (as.numeric(ednaoverlaptab) / as.numeric(ednaknowntab)))

colnames(test) = c("LKTL",
                   "InImage",
                   "ImageTP",
                   "ImageFN",
                   "Recall",
                   "InEDNA",
                   "eDNATP",
                   "eDNAFN",
                   "Precision")
