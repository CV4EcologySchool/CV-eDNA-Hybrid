library(cowplot)
library(vegan)
library(ggplot2)
library(indicspecies)
library (sciplot)

##fixed "april"
##fixed group names##

summ.tile <- read.csv("tiledatafull2.csv", header = T)
str(summ.tile)
# make a dataset of just the percent cover of the tiles, no other columns
##added in matrix command
summ.cover <- as.matrix(summ.tile[,5:18])
# make it a matrix with the data square root transformed
summ.tile.mat <- sqrt(summ.cover)

# make a dissimilarity matrix using "bray" method
summ.tile.dist<-vegdist(test, method='jaccard', binary = T)

#set seed to randomize 
set.seed(36) #reproducible results, can be any number

# This is the PerMANOVA code for the analysis
# use the dissimilarity matrix with the sqrt transformed data and test
# the percent cover of each category by plot type and in order of date (i.e. does it change over time)
summ.tile.div<-adonis2(summ.tile.dist~event_names, permutations = 999, method="jaccard")
summ.tile.div
# We have a strong interaction between plot type and date!!!
# try switching the order of variables and see if the results are different
summ.tile.div2<-adonis2(summ.tile.dist~date2*plot.type, data=summ.tile, permutations = 999, method="bray", strata="PLOT")
summ.tile.div2
# results are very similar

# PCA plot 
summ.dispersion<-betadisper(summ.tile.dist, group=event_names)
permutest(summ.dispersion)
plot(summ.dispersion, hull=FALSE, ellipse=TRUE) ##sd ellipse

### now nmds ###

summ.tile.MDS<-metaMDS(test, distance="jaccard", k=2, trymax=35, autotransform=TRUE) ##k is the number of dimensions
summ.tile.MDS # multivariate dispersians are homogenious 
stressplot(summ.tile.MDS)


##ok now follow
# https://chrischizinski.github.io/rstats/vegan-ggplot2/
  
data.scores <- as.data.frame(scores(summ.tile.MDS))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
data.scores$event <- event_names # create a column of site names, from the rownames of data.scores
data.scores$grp <- summ.tile$plot.type  #  add the grp variable created earlier
head(data.scores)  #look at the data

data.scores$date <- summ.tile$date2



species.scores <- as.data.frame(scores(summ.tile.MDS, "species"))  #Using the scores function from vegan to extract the species scores and convert to a data.frame
species.scores$species <- rownames(species.scores)  # create a column of species, from the rownames of species.scores
head(species.scores)  #look at the data

ggplot() + 
  geom_text(data=species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=data.scores,aes(x=NMDS1,y=NMDS2),size=2) + # add the point markers
  # geom_text(data=data.scores,aes(x=NMDS1,y=NMDS2,label=site),size=6,vjust=0) +  # add the site labels
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  coord_equal() +
  theme_bw()

ggplot() + 
  geom_text(data=species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=2) + # add the point markers
  geom_text(data=data.scores,aes(x=NMDS1,y=NMDS2,label=site),size=4,vjust=0,hjust=0) +  # add the site labels
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())

grp.open <- data.scores[data.scores$grp == "open", ][chull(data.scores[data.scores$grp == 
                                                                   "open", c("NMDS1", "NMDS2")]), ]  # hull values for grp A
grp.cage <- data.scores[data.scores$grp == "cage", ][chull(data.scores[data.scores$grp == 
                                                                   "cage", c("NMDS1", "NMDS2")]), ]  # hull values for grp B
grp.pc <- data.scores[data.scores$grp == "pc", ][chull(data.scores[data.scores$grp == 
                                                                         "pc", c("NMDS1", "NMDS2")]), ]  # hull values for grp B

hull.data <- rbind(grp.open, grp.cage,grp.pc)  #combine grp.a and grp.b
hull.data

ggplot() + 
  geom_polygon(data=hull.data,aes(x=NMDS1,y=NMDS2,fill=grp,group=grp),alpha=0.30) + # add the convex hulls
  geom_text(data=species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=4) + # add the point markers
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  scale_fill_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  scale_shape_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  scale_colour_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())+
  theme(legend.text = element_text(size = 12))+
  theme(legend.title = element_text(size = 14))

p1 <- ggplot() + 
  geom_polygon(data=hull.data,aes(x=NMDS1,y=NMDS2,fill=grp,group=grp),alpha=0.30) + # add the convex hulls
  geom_text(data=species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=4) + # add the point markers
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  scale_fill_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  scale_shape_discrete(name="Urchin treatment",
                       breaks=c("cage", "open", "pc"),
                       labels=c("excluded","present","partial cage"))+
  scale_colour_discrete(name="Urchin treatment",
                        breaks=c("cage", "open", "pc"),
                        labels=c("excluded","present","partial cage"))+
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())+
  theme(legend.text = element_text(size = 12))+
  theme(legend.title = element_text(size = 14))

  

######################################################################################
###                           WINTER NMDS PLOT                                     ###
######################################################################################

win.tile <- read.csv("win.tile.all.items.without.plot1.csv", header = T)
str(win.tile)

# make a dataset of just the percent cover of the tiles, no other columns
##added in matrix command
win.cover <- as.matrix(win.tile[,5:18])
# make it a matrix with the data square root transformed
win.tile.mat <- sqrt(win.cover)

# make a dissimilarity matrix using "bray" method
win.tile.dist<-vegdist(win.tile.mat, method='bray')

#set seed to randomize 
set.seed(36) #reproducible results, can be any number

# This is the PerMANOVA code for the analysis
# use the dissimilarity matrix with the sqrt transformed data and test
# the percent cover of each category by plot type and in order of date (i.e. does it change over time)
win.tile.div<-adonis2(win.tile.dist~plot.type*date2, data=win.tile, permutations = 999, method="bray", strata="PLOT")
win.tile.div
# We have a strong interaction between plot type and date!!!

# PCA plot 
win.dispersion<-betadisper(win.tile.dist, group=win.tile$plot.type)
permutest(win.dispersion)
plot(win.dispersion, hull=FALSE, ellipse=TRUE) ##sd ellipse

### now nmds ###

win.tile.MDS<-metaMDS(win.tile.mat, distance="bray", k=2, trymax=35, autotransform=TRUE) ##k is the number of dimensions
win.tile.MDS # multivariate dispersians are homogenious 
stressplot(win.tile.MDS)


##ok now follow
# https://chrischizinski.github.io/rstats/vegan-ggplot2/

w.data.scores <- as.data.frame(scores(win.tile.MDS))  #Using the scores function from vegan to extract the site scores and convert to a data.frame
w.data.scores$site <- rownames(w.data.scores)  # create a column of site names, from the rownames of data.scores
w.data.scores$grp <- win.tile$plot.type  #  add the grp variable created earlier
head(w.data.scores)  #look at the data

w.species.scores <- as.data.frame(scores(win.tile.MDS, "species"))  #Using the scores function from vegan to extract the species scores and convert to a data.frame
w.species.scores$species <- rownames(w.species.scores)  # create a column of species, from the rownames of species.scores
head(w.species.scores)  #look at the data

ggplot() + 
  geom_text(data=w.species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=w.data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=2) + # add the point markers
  geom_text(data=w.data.scores,aes(x=NMDS1,y=NMDS2,label=site),size=6,vjust=0) +  # add the site labels
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  coord_equal() +
  theme_bw()

ggplot() + 
  geom_text(data=w.species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=w.data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=2) + # add the point markers
  geom_text(data=w.data.scores,aes(x=NMDS1,y=NMDS2,label=site),size=4,vjust=0,hjust=0) +  # add the site labels
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())

w.grp.open <- w.data.scores[w.data.scores$grp == "open", ][chull(w.data.scores[w.data.scores$grp == 
                                                                         "open", c("NMDS1", "NMDS2")]), ]  # hull values for grp A
w.grp.cage <- w.data.scores[w.data.scores$grp == "cage", ][chull(w.data.scores[w.data.scores$grp == 
                                                                         "cage", c("NMDS1", "NMDS2")]), ]  # hull values for grp B
w.grp.pc <- w.data.scores[w.data.scores$grp == "pc", ][chull(w.data.scores[w.data.scores$grp == 
                                                                     "pc", c("NMDS1", "NMDS2")]), ]  # hull values for grp B

w.hull.data <- rbind(w.grp.open, w.grp.cage, w.grp.pc)  #combine grp.a and grp.b
w.hull.data

ggplot() + 
  geom_polygon(data=w.hull.data,aes(x=NMDS1,y=NMDS2,fill=grp,group=grp),alpha=0.30) + # add the convex hulls
  geom_text(data=w.species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=w.data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=4) + # add the point markers
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  scale_fill_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  scale_shape_discrete(name="Urchin treatment",
                       breaks=c("cage", "open", "pc"),
                       labels=c("excluded","present","partial cage"))+
  scale_colour_discrete(name="Urchin treatment",
                        breaks=c("cage", "open", "pc"),
                        labels=c("excluded","present","partial cage"))+
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())+
  theme(legend.text = element_text(size = 12))+
  theme(legend.title = element_text(size = 14))

p2 <- ggplot() + 
  geom_polygon(data=w.hull.data,aes(x=NMDS1,y=NMDS2,fill=grp,group=grp),alpha=0.30) + # add the convex hulls
  geom_text(data=w.species.scores,aes(x=NMDS1,y=NMDS2,label=species),alpha=0.5) +  # add the species labels
  geom_point(data=w.data.scores,aes(x=NMDS1,y=NMDS2,shape=grp,colour=grp),size=4) + # add the point markers
  scale_colour_manual(values=c("cage" = "red", "open" = "green","pc" = "blue")) +
  scale_fill_discrete(name="Urchin treatment",
                      breaks=c("cage", "open", "pc"),
                      labels=c("excluded","present","partial cage"))+
  scale_shape_discrete(name="Urchin treatment",
                       breaks=c("cage", "open", "pc"),
                       labels=c("excluded","present","partial cage"))+
  scale_colour_discrete(name="Urchin treatment",
                        breaks=c("cage", "open", "pc"),
                        labels=c("excluded","present","partial cage"))+
  coord_equal() +
  theme_bw() + 
  theme(axis.text.x = element_blank(),  # remove x-axis text
        axis.text.y = element_blank(), # remove y-axis text
        axis.ticks = element_blank(),  # remove axis ticks
        axis.title.x = element_text(size=18), # remove x-axis labels
        axis.title.y = element_text(size=18), # remove y-axis labels
        panel.background = element_blank(), 
        panel.grid.major = element_blank(),  #remove major-grid labels
        panel.grid.minor = element_blank(),  #remove minor-grid labels
        plot.background = element_blank())+
  theme(legend.text = element_text(size = 12))+
  theme(legend.title = element_text(size = 14))


###########################################################################
###                       Combine plots                                 ###
###########################################################################

plot_grid(p1, p2,
          label_x = 0, label_y = 1, nrow=2, ncol=1, align="h",
          labels = c("Summer","Winter"), hjust=-0.5,
          label_size=20,
          rel_widths = c(4,4))
