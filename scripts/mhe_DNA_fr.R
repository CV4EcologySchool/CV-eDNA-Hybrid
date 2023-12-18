dna_LKTL_Long = factor(dna_LKTL_Long, levels = levels(invert_cleanlab$LKTL_Long))

one_hot_encoded <- df %>%
  pivot_wider(names_from = dna_LKTL_Long, values_from = dna_LKTL_Long, values_fn = length, values_fill = 0) %>%
  mutate_if(is.numeric, ~ifelse(. > 0, 1, 0))

missing_levels <- setdiff(levels(dna_LKTL_Long), colnames(one_hot_encoded))

# Add missing columns filled with 0s
for (missing_level in missing_levels) {
  one_hot_encoded[[missing_level]] <- 0
}


one_hot_encoded <- one_hot_encoded %>%
  select(edna_rmdup.Event, order(names(one_hot_encoded))) %>%
  arrange(edna_rmdup.Event)


