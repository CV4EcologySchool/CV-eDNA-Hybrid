filesdf = paste(invertmatch$Label,invertmatch$ROI,"jpg", sep = ".")
filesdftif = paste(invertmatch$Label, "tif", invertmatch$ROI, "jpg", sep = ".")
filesdftif01 = paste0(substr(filesdftif, 1, 20), substr(filesdftif, 24, str_length(filesdftif)))
filesdftif01ib = paste0(substr(filesdftif, 1, 17), ".ib", substr(filesdftif, 21, str_length(filesdftif)))
filesdftif01s = paste0(substr(filesdftif, 1, 22), substr(filesdftif, 26, str_length(filesdftif)))

filecombine = list(filesdf,
                   filesdftif,
                   filesdftif01,
                   filesdftif01ib,
                   filesdftif01s)

for(config in filecombine){
  invertmatch$Label[config %in% images] = config[config %in% images]
}




strange = Label[which(filesdf %!in% images 
                & filesdftif %!in% images
                & filesdf01 %!in% images
                & filesdftif01s %!in% images)]
table(substr(strange, 1, 17))
