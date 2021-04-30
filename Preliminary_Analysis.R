# Preliminary_Analysis.R
# Do some summary statistics on VDem

library(tidyverse)

# Start with a very small sample read-in for speed
VDem <- read_csv('~/../Box/ECMA-31330-Project/V-Dem-CY-Full+Others-v11.1.csv', n_max = 1000)

# Summaries <- VDem %>% 
#     summarize_all(mean) %>%
#     print(n=Inf)

# OLS of political instability on turnout

# Here is the WB politcal stability index. Higher = better
# summary_table <- VDem %>%
#     select(e_wbgi_pve) %>%
#     summarize(instab_mean = mean(e_wbgi_pve), instab_median = median(e_wbgi_pve), instab_n = n(e_wbgi_pve), instab_sd = sd(e_wbgi_pve))

# Turnout variable v2eltrnout

OLS_instab_turnout <- lm(e_wbgi_pve ~ v2eltrnout, data = VDem)
print(OLS_instab_turnout)
