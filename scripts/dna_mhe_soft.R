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

