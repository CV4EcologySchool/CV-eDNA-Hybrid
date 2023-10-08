library(tidyr)
library(stringr)

taxaorder = str_to_title(taxaorder)
# taxaorder = c("Species", taxaorder)
flow = taxaorder
flow = flow[-c(6,7,9,12)]
flow = data.frame(flow)
flow[,c(2:14)] = 0
colnames(flow) = c("OG", taxaorder)


for(i in 1:nrow(flow)){
  for(j in 1:length(taxaorder)){
    flow[i,j+1] = sum(splevels[splevelsog == flow[i,1]] == taxaorder[j])
  }
}

tidyflow = gather(flow, end, amount, -OG)

OG = flow$OG
nodes <- data.frame(node = c(0:(sum(nrow(flow), (ncol(flow) - 1))-1)), 
                    name = c(OG, str_to_lower(taxaorder)))
#create links dataframe
tidyflow$end = str_to_lower(tidyflow$end)
tidyflow <- merge(tidyflow, nodes, by.x = "OG", by.y = "name")
tidyflow <- merge(tidyflow, nodes, by.x = "end", by.y = "name")
links <- tidyflow[ , c("node.x", "node.y", "amount")]
colnames(links) <- c("source", "target", "value")

links$group = paste0("group", links$source)
nodes$group = nodes$name
nodes$name = str_to_title(nodes$name)
links = links[links$value > 0,]
my_color <- 'd3.scaleOrdinal() 
.domain(["group0", "group1", "group2", "group3", "group4", "group5", "group6", "group7", "group8",
"Species", "Genus", "Subfamily", "Family", "Superfamily", "Infraorder", "Suborder", "Order", "Superorder", "Subclass", "Class", "Subphylum", "Phylum"])
.range(["#fde725", "#addc30", "#5ec962", "#28ae80", "#21918c", "#2c728e", "#3b528b", "#472d7b", "#440154",
"#fde725", "#c8e020", "#90d743", "#5ec962", "#35b779", "#20a486", "#21918c", "#287c8e", "#31688e", "#3b528b", "#443983", "#481f70", "#440154",
"#fde725", "#addc30", "#5ec962", "#28ae80", "#21918c", "#2c728e", "#3b528b", "#472d7b", "#440154"])'


networkD3::sankeyNetwork(Links = links, Nodes = nodes, 
                         Source = 'source', 
                         Target = 'target', 
                         Value = 'value', 
                         NodeID = 'name',
                         units = 'amount',
                         fontSize = 16,
                         colourScale = my_color,
                         LinkGroup = 'group',
                         iterations = 0)


