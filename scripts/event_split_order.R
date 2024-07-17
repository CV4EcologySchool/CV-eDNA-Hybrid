library(caret)
# Sampling by event

events = as.data.frame(table(invert_cleanlab$Event))

set.seed(123)
shuffled_data <- events[sample(nrow(events)), ]

# Initialize variables for tracking
cumulative_freq <- 0
sampled_events <- character(0)
stop = nrow(invert_cleanlab) * 0.15

# holdout_events = c("OAES00520160629",
#                    "CPER00220160714")
# shuffled_data = shuffled_data[-which(shuffled_data$Var1 %in% holdout_events),]

set.seed(123)
# Iterate through shuffled rows
for (i in 1:nrow(shuffled_data)) {
  site <- as.character(shuffled_data[i, "Var1"])
  freq <- shuffled_data[i, "Freq"]
  
  # Check if adding the current frequency exceeds 100
  if (cumulative_freq + freq >= stop) {
    sampled_events <- c(sampled_events, site)
    break
  } else {
    sampled_events <- c(sampled_events, site)
    cumulative_freq <- cumulative_freq + freq
  }
}

invert_cleanlab$order_plus = as.factor(invert_cleanlab$order_plus)
invert_cleanlab$longlab = as.factor(invert_cleanlab$longlab)
invert_cleanlab$Label = paste0(invert_cleanlab$Label, ".tif.", invert_cleanlab$ROI, ".jpg")

val_split1 = invert_cleanlab[which(invert_cleanlab$Event %in% sampled_events),]
train_split1 = invert_cleanlab[-which(invert_cleanlab$Event %in% sampled_events),]



#######################################################################################
val_split1_abund = as.data.frame(table(val_split1$order_plus))
val_split1_prev = as.data.frame(table(val_split1$order_plus)/nrow(val_split1))
train_split1_abund = as.data.frame(table(train_split1$order_plus))
train_split1_prev = as.data.frame(table(train_split1$order_plus)/nrow(train_split1))

prev_combined = cbind(train_split1_prev, val_split1_prev)
prev_combined = prev_combined[,-3]

prev_combined$ratio = prev_combined$Freq/prev_combined$Freq.1

abund_combined = cbind(train_split1_abund, val_split1_abund)
abund_combined = abund_combined[,-3]

abund_combined$ratio = abund_combined$Freq/abund_combined$Freq.1


mhe = read.csv("C:/Users/jarre/ownCloud/CV-eDNA/dna_mhe_order.csv")
mhe = mhe[,-1]
colnames(mhe) = c(levels(as.factor((invert_cleanlab$longlab))))
event_names = events$Var1
mhe$Event = event_names

train_split1 = merge(train_split1, mhe, by = "Event", all = F)
val_split1 = merge(val_split1, mhe, by = "Event", all = F)

write.csv(train_split1, "train.csv", row.names = F)
write.csv(val_split1, "valid.csv", row.names = F)


dna_train = edna_rmdup[which(edna_rmdup$event %!in% sampled_events),]
dna_train_LKTL_Long = dna_LKTL_Long[which(edna_rmdup$event %!in% sampled_events)]
