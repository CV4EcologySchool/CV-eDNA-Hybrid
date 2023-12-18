library(cowplot)
library(vegan)
library(ggplot2)
library(indicspecies)
library (sciplot)
library(dplyr)
library(lubridate)
library(tidyr)

setwd("C:/Carabid_Data/CV-eDNA/splits/LKTL-37141")

##fixed "april"
##fixed group names##

# dna_mhe = read.csv("naive.csv")
# image_mhe = read.csv("image_mhe.csv")
# colnames(dna_mhe)[1] = "event"
# colnames(image_mhe) = colnames(dna_mhe)
# dna_mhe$source = "DNA"
# image_mhe$source = "Morph"
# full_mhe = rbind(image_mhe, dna_mhe)
# full_mhe <- full_mhe %>%
#   select(ncol(full_mhe), everything())

ydays = as.character(as.Date(substr(events, 
                                            nchar(events) - 7, 
                                            nchar(events)),
                                     format = "%Y%m%d"))
ydays = yday(ydays)

meta <- invert_cleanlab %>%
  group_by(Event) %>%
  summarise(lat = mean(decLat), long = mean(decLong), elevM = mean(elevM))

meta$yday = ydays
meta$habitat = unique(invert_cleanlab[, c("Event", "nlcdClass")])[,2]
meta$site = substr(meta$Event, 1, 4)

wide_dna <- edna_rmdup %>%
  group_by(Event, LITL) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = LITL, values_from = count, values_fill = 0)

wide_mike = invert_cleanlab %>%
  group_by(Event, AllTaxa) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = AllTaxa, values_from = count, values_fill = 0)

wide_mike_LKTL = invert_cleanlab %>%
  group_by(Event, LKTL) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = LKTL, values_from = count, values_fill = 0)


ml_class = data.frame(val_split1$Event, newclass)
colnames(ml_class) = c("Event", "Class")
wide_ml = ml_class %>%
  group_by(Event, Class) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = Class, values_from = count, values_fill = 0)

hybrid_class = data.frame(val_split1$Event, mrca)
colnames(hybrid_class) = c("Event", "Class")
wide_hybrid = hybrid_class %>%
  group_by(Event, Class) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = Class, values_from = count, values_fill = 0)

wide_mike_val = val_split1 %>%
  group_by(Event, AllTaxa) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = AllTaxa, values_from = count, values_fill = 0)

wide_dna_val = edna_rmdup[which(edna_rmdup$Event %in% wide_ml$Event),] %>%
  group_by(Event, LITL) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = LITL, values_from = count, values_fill = 0)

wide_mike_val_LKTL = val_split1 %>%
  group_by(Event, LKTL) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = LKTL, values_from = count, values_fill = 0) 

hybrid_class = data.frame(val_split1$Event, mrca)
colnames(hybrid_class) = c("Event", "Class")
wide_mike_val_spec = hybrid_class %>%
  group_by(Event, Class) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = Class, values_from = count, values_fill = 0)

spatio = read.csv("pred_spatio.csv")
spatio = spatio[,1]
spatio_class = data.frame(val_split1$Event, spatio)
colnames(spatio_class) = c("Event", "Class")
wide_spatio = spatio_class %>%
  group_by(Event, Class) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = Class, values_from = count, values_fill = 0)

image_only = read.csv("preds_unimodal.csv")
image_only = image_only[,1]
image_class = data.frame(val_split1$Event, image_only)
colnames(image_class) = c("Event", "Class")
wide_image = image_class %>%
  group_by(Event, Class) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = Class, values_from = count, values_fill = 0)


# meta
ml_meta = meta[which(meta$Event %in% wide_ml$Event),]

str(full_mhe)

nmds_plot = function(mhe, meta, name = "N/A", binary = F){
  # mhe is one of the wide data frames created above 
  # make a dataset of just the percent cover of the tiles, no other columns
  ##added in matrix command
  mhe_mat <- as.matrix(mhe[,-1])

  #set seed to randomize 
  set.seed(36) #reproducible results, can be any number
  

  ### now nmds ###
  
  mhe.MDS<-metaMDS(mhe_mat, distance="jaccard", k=2, trymax=35, autotransform=TRUE, binary = binary) ##k is the number of dimensions
  mhe.MDS # multivariate dispersians are homogenious 
  stressplot(mhe.MDS, main = name)
  
  
  ##ok now follow
  # https://chrischizinski.github.io/rstats/vegan-ggplot2/
    
  meta = cbind(meta, scores(mhe.MDS, "sites"))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
  #data.scores$site <- rownames(data.scores)# create a column of site names, from the rownames of data.scores
  
  cnames = colnames(meta)
  
  meta = within(meta, {
    lat <- scale(lat)
    long <- scale(long)
    elevM <- scale(elevM)
    yday = scale(yday)
  })
  
  colnames(meta) = cnames
  
  model1 = lmer(NMDS1 ~ lat * long + elevM + habitat + (1|site/yday), data = meta)
  model2 = lmer(NMDS2 ~ lat * long + elevM + habitat + (1|site/yday), data = meta)

  lat1.plot = ggplot() + 
    geom_point(data=meta,aes(x=lat,y=NMDS1),size=2)
  
  lat2.plot = ggplot() + 
    geom_point(data=meta,aes(x=lat,y=NMDS2),size=2)
  
  nmds.plot = ggplot() + 
    geom_point(data=meta,aes(x=NMDS1,y=NMDS2),size=2)

  return(list(lat1.plot = lat1.plot, 
              lat2.plot = lat2.plot,
              nmds.plot = nmds.plot, 
              model1.sum = summary(model1, correlation = T), 
              model2.sum = summary(model2, correlation = T),
              mds = mhe.MDS,
              data = meta))
}

rm_row = which(rowSums(wide_dna[,-1]) < 5)

results_dna = nmds_plot(wide_dna[-rm_row,], meta = meta[-rm_row,], binary = T, name = "DNA Full")
results_mike = nmds_plot(wide_mike[-rm_row,], meta = meta[-rm_row,], name = "Mike Full")
results_mike_LKTL = nmds_plot(wide_mike_LKTL[-rm_row,], meta = meta[-rm_row,], name = "Mike Full LKTL")

results_mike_val = nmds_plot(wide_mike_val[-2,], meta = ml_meta[-2,], name = "Mike val")
results_ml = nmds_plot(wide_ml[-2,], meta = ml_meta[-2,], name = "ML")
results_hybrid = nmds_plot(wide_hybrid[-2,], meta = ml_meta[-2,], name = "ML Specificity Filter")
results_dna_val = nmds_plot(wide_dna_val[-2,], meta = ml_meta[-2,], binary = T, name = "DNA val")
results_mike_val_LKTL = nmds_plot(wide_mike_val_LKTL[-2,], meta = ml_meta[-2,], name = "Mike val LKTL")
results_mike_val_spec = nmds_plot(wide_mike_val_spec[-2,], meta = ml_meta[-2,], name = "Mike val Spec")

results_dna$model1.sum
results_mike$model1.sum
results_mike_LKTL$model1.sum

results_dna_val$model1.sum
results_mike_val$model1.sum
results_mike_val_LKTL$model1.sum
results_mike_val_spec$model1.sum
results_ml$model1.sum
results_hybrid$model1.sum


results_dna$model2.sum
results_mike$model2.sum
results_mike_LKTL$model2.sum

results_dna_val$model2.sum
results_mike_val$model2.sum
results_mike_val_LKTL$model2.sum
results_mike_val_spec$model2.sum
results_ml$model2.sum
results_hybrid$model2.sum


wide_ml$source = "DNA"
wide_image$source = "Image_Only"
wide_spatio$source = "Spatiotemporal"
wide_mike_val_LKTL$source = "Ground_Truth"


bruh = rbind(wide_image, wide_ml, wide_spatio, wide_mike_val_LKTL)
bruh <- bruh %>%
  select(source, everything()) %>%
  mutate_all(~ifelse(is.na(.), 0, .))

set.seed(123)

test = pairwise.adonis2(mhe_mat ~ source, method = "bray", data = bruh)

meta = rbind(ml_meta, ml_meta, ml_meta, ml_meta)


mhe_mat <- as.matrix(bruh[,-c(1,2)])

#set seed to randomize 
set.seed(36) #reproducible results, can be any number


### now nmds ###

mhe.MDS<-metaMDS(mhe_mat, distance="bray", k=2, trymax=35, autotransform=TRUE, binary = binary) ##k is the number of dimensions
mhe.MDS # multivariate dispersians are homogenious 
stressplot(mhe.MDS, main = "Bray")


##ok now follow
# https://chrischizinski.github.io/rstats/vegan-ggplot2/

meta = cbind(meta, scores(mhe.MDS, "sites"))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
#data.scores$site <- rownames(data.scores)# create a column of site names, from the rownames of data.scores

cnames = colnames(meta)

meta = within(meta, {
  lat <- scale(lat)
  long <- scale(long)
  elevM <- scale(elevM)
  yday = scale(yday)
})

colnames(meta) = cnames

meta$source = bruh$source

model1 = lmer(NMDS1 ~ lat * long + elevM + habitat + (1|site/yday) + source:lat, data = meta)
model2 = lmer(NMDS2 ~ lat * long + elevM + habitat + (1|site/yday), data = meta)

lat1.plot = ggplot() + 
  geom_point(data=meta,aes(x=lat,y=NMDS1),size=2)

lat2.plot = ggplot() + 
  geom_point(data=meta,aes(x=lat,y=NMDS2),size=2)

nmds.plot = ggplot() + 
  geom_point(data=meta,aes(x=NMDS1,y=NMDS2),size=2)


ggplot() + 
  geom_point(data=meta,aes(x=NMDS1,y=NMDS2,shape=source,colour=source),size=2) 


