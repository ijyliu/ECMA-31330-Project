# Preliminary_Analysis.R
# Do some summary statistics on VDem

library(pacman)
p_load('tidyverse', 'stargazer', 'estimatr')

# Start with a very small sample read-in for speed
VDem <- read_csv('~/../Box/ECMA-31330-Project/V-Dem-CY-Full+Others-v11.1.csv')

# Summaries <- VDem %>% 
#     summarize_all(mean) %>%
#     print(n=Inf)

# OLS of political instability on turnout

# Here is the WB politcal stability index. Higher = more stable
# summary_table <- VDem %>%
#     select(e_wbgi_pve) %>%
#     summarize(instab_mean = mean(e_wbgi_pve), instab_median = median(e_wbgi_pve), instab_n = n(e_wbgi_pve), instab_sd = sd(e_wbgi_pve))

# Turnout variable v2eltrnout

OLS_instab_turnout <- lm(e_wbgi_pve ~ v2eltrnout, data = VDem)
stargazer(OLS_instab_turnout, se = starprep(OLS_instab_turnout, se_type = "stata"), title="Regression of WB Political Stability on Election Voter Turnout", covariate.labels = c("Turnout"), dep.var.labels   = "Stability", dep.var.caption = "", align=TRUE, out="~/../repo/ECMA-31330-Project/Output/Instab_Turnout_Reg.tex", omit.stat=c("f", "ser", "adj.rsq"), notes = "Robust standard errors in parentheses.")
