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
weightedClass = read.csv("preds_weighted_named.csv")[,1]
fusionClass = read.csv("preds_concat_named.csv")[,1]
simmaskClass = read.csv("preds_simmask_named.csv")[,1]
simfusionClass = read.csv("preds_simfusion_named.csv")[,1]
noiseClass = read.csv("preds_noise_named.csv")[,1]

assemblBase = assemblClass(baseClass, 'Base')
assemblNaive = assemblClass(naiveClass, 'Naive mask')
assemblWeighted = assemblClass(weightedClass, 'Weighted mask')
assemblFusion = assemblClass(fusionClass, 'Fusion')
assemblSimmask = assemblClass(simmaskClass, 'Image mask')
assemblSimfusion = assemblClass(simfusionClass, 'Image fusion')
assemblNoise = assemblClass(noiseClass, 'Noise')


wide_mike_val_LKTL$source = "Ground truth"
assemblTrue = wide_mike_val_LKTL

assemblCombine = rbind(assemblTrue,
                       assemblBase,
                       assemblNaive,
                       assemblWeighted,
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

# meta = rbind(ml_meta, ml_meta, ml_meta, ml_meta, ml_meta, ml_meta)
# 
# 
# #set seed to randomize 
# set.seed(123) #reproducible results, can be any number
# 
# 
# ##ok now follow
# # https://chrischizinski.github.io/rstats/vegan-ggplot2/
# 
# meta = cbind(meta, scores(mhe.MDS, "sites"))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
# #data.scores$site <- rownames(data.scores)# create a column of site names, from the rownames of data.scores
# 
# # cnames = colnames(meta)
# # 
# # meta = within(meta, {
# #   lat <- scale(lat)
# #   long <- scale(long)
# #   elevM <- scale(elevM)
# #   yday = scale(yday)
# # })
# # 
# # colnames(meta) = cnames
# 
# meta$source = assemblCombine$source

distance_matrix <- vegdist(mhe_mat, method = "jaccard")
distance_matrix_full <- as.matrix(distance_matrix)

distance_matrix_comp = matrix(data = NA, nrow = 28, ncol = 7)
for(i in 1:28){
  for(j in 1:7){
    distance_matrix_comp[i,j] = distance_matrix_full[((28*j) + i), i]
  }
}
colnames(distance_matrix_comp) = unique(assemblCombine$source)[2:8]
rownames(distance_matrix_comp) = unique(assemblCombine$Event)


m_dist = melt(distance_matrix_comp)
m_dist %>%
  group_by(Var2) %>%
  identify_outliers(value)

m_dist %>%
  group_by(Var2) %>%
  shapiro_test(value)

res.aov <- anova_test(data = m_dist, dv = value, wid = Var1, within = Var2)
get_anova_table(res.aov)

pwc <- m_dist %>%
  pairwise_t_test(
    value ~ Var2, paired = TRUE,
    p.adjust.method = "BH",
    detailed = T
  )
pwc

bxp <- ggboxplot(m_dist, x = "Var2", y = "value")
bxp

pwc <- pwc %>% add_xy_position(x = "Var2")
bxp + 
  stat_pvalue_manual(pwc, hide.ns = T)+
  labs(
    subtitle = get_test_label(res.aov, detailed = TRUE),
    caption = get_pwc_label(pwc)
  )




set.seed(123)

permaov = pairwise.adonis2(mhe_mat ~ source, method = "jaccard", data = assemblCombine)

mhe.MDS<-metaMDS(mhe_mat, distance="jaccard", k=2, trymax=1000, autotransform=TRUE, binary = F) ##k is the number of dimensions
mhe.MDS # multivariate dispersians are homogenious 
stressplot(mhe.MDS, main = "Combined nMDS")

nMDSdf = cbind(assemblCombine, scores(mhe.MDS, "sites"))

nMDSdf$source = factor(nMDSdf$source, levels = c("Ground truth",
                                                 'Base',
                                                 'Naive mask',
                                                 'Weighted mask',
                                                 'Fusion',
                                                 'Image mask',
                                                 'Image fusion',
                                                 'Noise'))

#Plot code adapted from Emmons et al., 2023 (https://doi.org/10.1002/edn3.453)
centroid = nMDSdf %>% 
  group_by(source) %>% 
  summarize(NMDS1 = mean(NMDS1), NMDS2 = mean(NMDS2))


nmdsplot = ggplot(data = nMDSdf, aes(x=NMDS1, y=NMDS2, color = source)) +
    #geom_point(aes(color = source)) +
    stat_ellipse() + 
    geom_point(data = centroid, size = 3, shape = 24, color = "black", aes(fill = source), show.legend = F) + # add centroids
    scale_color_manual(values = c("#1C86EE", "#E31A1C", "#008B00", "#6A3D9A", "#FF7F00", "#000000", "#FFD700", "#B3B3B3")) +
    scale_fill_manual(values = c("#1C86EE", "#E31A1C", "#008B00", "#6A3D9A", "#FF7F00", "#000000", "#FFD700", "#B3B3B3")) +
    xlim(-1.5, 2) +
    theme_classic() +
    theme(legend.text=element_text(size=12),
          legend.title = element_blank(),
          axis.text.x = element_text(color = "black", size = 10),
          axis.text.y = element_text(color = "black", size = 10),
          axis.title.x = element_text(color = "black", size = 12),
          axis.title.y = element_text(color = "black", size = 12),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank())

ggsave("nmds_nodots.png", plot = nmdsplot, dpi = 300, width = 10, height = 8, units = "in")

