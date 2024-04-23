library(cowplot)
library(vegan)
library(ggplot2)
library(indicspecies)
library (sciplot)
library(dplyr)
library(lubridate)
library(tidyr)
library(lmerTest)
library(pairwiseAdonis)
library(tidyr)

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

ml_meta = meta[which(meta$Event %in% wide_ml$Event),]



wide_mike_val_LKTL = val_split1 %>%
  group_by(Event, LKTL) %>%
  summarise(count = n()) %>%
  pivot_wider(names_from = LKTL, values_from = count, values_fill = 0) 

# Turn classifications into assemblage data
assemblClass = function(x, name){
  ml_class = data.frame(val_split1$Event, x)
  colnames(ml_class) = c("Event", "Class")
  wide_ml = ml_class %>%
    group_by(Event, Class) %>%
    summarise(count = n()) %>%
    pivot_wider(names_from = Class, values_from = count, values_fill = 0)
  
  wide_ml$source = name
  return(wide_ml)
}

baseClass = read.csv("preds_baseline_named.csv")[,1]
naiveClass = read.csv("preds_naive_named.csv")[,1]
fusionClass = read.csv("preds_concat_named.csv")[,1]
simmaskClass = read.csv("preds_simmask_named.csv")[,1]
simfusionClass = read.csv("preds_simfusion_named.csv")[,1]
noiseClass = read.csv("preds_noise_named.csv")[,1]

assemblBase = assemblClass(baseClass, 'Base')
assemblNaive = assemblClass(naiveClass, 'Naive mask')
assemblFusion = assemblClass(fusionClass, 'Fusion')
assemblSimmask = assemblClass(simmaskClass, 'Image mask')
assemblSimfusion = assemblClass(simfusionClass, 'Image fusion')
assemblNoise = assemblClass(noiseClass, 'Noise')


wide_mike_val_LKTL$source = "Ground_Truth"
assemblTrue = wide_mike_val_LKTL

assemblCombine = rbind(assemblTrue,
                       assemblBase,
                       # assemblNaive,
                       assemblFusion,
                       assemblSimmask,
                       assemblSimfusion,
                       assemblNoise)

assemblCombine <- assemblCombine %>%
  select(source, everything()) %>%
  mutate_all(~ifelse(is.na(.), 0, .))

mhe_mat <- as.matrix(assemblCombine[,-c(1,2)])

# set.seed(123)
# 
# test = pairwise.adonis2(mhe_mat ~ source, method = "jaccard", data = assemblCombine)

meta = rbind(ml_meta, ml_meta, ml_meta, ml_meta, ml_meta, ml_meta)


#set seed to randomize 
set.seed(123) #reproducible results, can be any number


### now nmds ###

mhe.MDS<-metaMDS(mhe_mat, distance="jaccard", k=2, trymax=1000, autotransform=TRUE) ##k is the number of dimensions
mhe.MDS # multivariate dispersians are homogenious 
stressplot(mhe.MDS, main = "Jaccard")


##ok now follow
# https://chrischizinski.github.io/rstats/vegan-ggplot2/

meta = cbind(meta, scores(mhe.MDS, "sites"))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
#data.scores$site <- rownames(data.scores)# create a column of site names, from the rownames of data.scores

# cnames = colnames(meta)
# 
# meta = within(meta, {
#   lat <- scale(lat)
#   long <- scale(long)
#   elevM <- scale(elevM)
#   yday = scale(yday)
# })
# 
# colnames(meta) = cnames

meta$source = assemblCombine$source

# model1 = lmer(NMDS1 ~ lat * long + elevM + habitat + (1|site/yday) + source:lat, data = meta)
# model2 = lmer(NMDS2 ~ lat * long + elevM + habitat + (1|site/yday), data = meta)
# 
# lat1.plot = ggplot() + 
#   geom_point(data=meta,aes(x=lat,y=NMDS1),size=2)
# 
# lat2.plot = ggplot() + 
#   geom_point(data=meta,aes(x=lat,y=NMDS2),size=2)
# 
# nmds.plot = ggplot() + 
#   geom_point(data=meta,aes(x=NMDS1,y=NMDS2),size=2)


ggplot() + 
  geom_point(data=meta,aes(x=NMDS1,y=NMDS2,shape=source,colour=source),size=2) 


NMDS1 = split(meta$NMDS1, meta$source)
NMDS2 = split(meta$NMDS2, meta$source)

sources = unique(meta$source)

par(mfrow = c(2, 3))
for(i in 1:5){
  plot(NMDS1[[i]] ~ NMDS1[[6]], main = sources[i])
  abline(a = 0, b = 1, col = "red")
  ss_res <- sum((NMDS1[[i]] - NMDS1[[6]])^2)
  ss_tot <- sum((NMDS1[[i]] - mean(NMDS1[[i]]))^2)
  r_squared <- 1 - ss_res / ss_tot
  legend("topleft", legend = sprintf("R^2 = %.2f", r_squared), bty = "n", cex = 1)
}

par(mfrow = c(2, 3))
for(i in 1:5){
  plot(NMDS2[[i]] ~ NMDS2[[6]], 
       main = sources[i])
  abline(a = 0, b = 1, col = "red")
  ss_res <- sum((NMDS2[[i]] - NMDS2[[6]])^2)
  ss_tot <- sum((NMDS2[[i]] - mean(NMDS2[[i]]))^2)
  r_squared <- 1 - ss_res / ss_tot
  legend("topleft", legend = sprintf("R^2 = %.2f", r_squared), bty = "n", cex = 1)
}

## Making a linear regression using the NMDS values as input, which is probably wrong
# lm_NMDS1 = list()
# for(i in 1:5){
#   lm_NMDS1[[i]] = list(sources[i], lm(NMDS1[[i]] ~ NMDS1[[6]]))
#   plot(NMDS1[[i]] ~ NMDS1[[6]], main = sources[i])
#   abline(lm_NMDS1[[i]][[2]], col = "red")
#   r_squared = summary(lm_NMDS1[[i]][[2]])$r.squared
#   legend("topleft", legend = sprintf("R^2 = %.2f", r_squared), bty = "n", cex = 0.8)
# }


distance_matrix <- vegdist(mhe_mat, method = "jaccard")
distance_matrix_full <- as.matrix(distance_matrix)

distance_matrix_comp = matrix(data = NA, nrow = 28, ncol = 5)
for(i in 1:28){
  for(j in 1:5){
    distance_matrix_comp[i,j] = distance_matrix_full[((28*j) + i), i]
  }
}
colnames(distance_matrix_comp) = unique(meta$source)[2:6]
rownames(distance_matrix_comp) = unique(meta$Event)


test = melt(distance_matrix_comp)
test %>%
  group_by(Var2) %>%
  identify_outliers(value)

test %>%
  group_by(Var2) %>%
  shapiro_test(value)

res.aov <- anova_test(data = test, dv = value, wid = Var1, within = Var2)
get_anova_table(res.aov)

pwc <- test %>%
  pairwise_t_test(
    value ~ Var2, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc

bxp <- ggboxplot(test, x = "Var2", y = "value")
bxp

pwc <- pwc %>% add_xy_position(x = "Var2")
bxp + 
  stat_pvalue_manual(pwc, hide.ns = T)+
  labs(
    subtitle = get_test_label(res.aov, detailed = TRUE),
    caption = get_pwc_label(pwc)
  )



