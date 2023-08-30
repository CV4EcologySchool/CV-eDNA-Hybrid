invertmatch$Det_Level[which(invertmatch$X == 16412)] = "Subfamily"
invertmatch$AllTaxa[which(invertmatch$X == 16412)] = "Entiminae"

invert_cleanlab = invertmatch[-which(invertmatch$AllTaxa == "Arthropoda"),]
invert_cleanlab$Class[which(invert_cleanlab$Subphylum == "Crustacea")] = "Crustacea"
ednamatch$class[which(ednamatch$subphylum == "Crustacea")] = "Crustacea"

invert_cleanlab = invert_cleanlab[-which(invert_cleanlab$AllTaxa == "Arachnida"),]

invert_cleanlab$Class[which(invert_cleanlab$Subphylum == "Myriapoda")] = "Myriapoda"

invert_cleanlab = invert_cleanlab[-which(invert_cleanlab$AllTaxa == "Insecta"),]
invert_cleanlab$Class[which(invert_cleanlab$Class == "Protura")] = "Insecta"

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

