library(dplyr)
library(purrr)

lktl_event_summary <- invert_cleanlab %>%
  group_by(LKTL) %>%
  summarize(Present_Events = paste(Event, collapse = ", ")) %>%
  ungroup()

species_site_dict <- lktl_event_summary %>%
  mutate(Present_Events = strsplit(Present_Events, ", ")) %>%
  pull(Present_Events) %>%
  set_names(lktl_event_summary$LKTL)

unique_lists <- sapply(species_site_dict, unique)
taxa_events <- sapply(unique_lists, length)
taxa_events = as.data.frame(taxa_events)
