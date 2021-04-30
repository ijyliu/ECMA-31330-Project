# Preliminary_Analysis.R
# Do some summary statistics on VDem

library(pacman)
p_load('tidyverse', 'stargazer', 'estimatr')

# Start with a very small sample read-in for speed
VDem <- read_csv('~/../Box/ECMA-31330-Project/V-Dem-CY-Full+Others-v11.1.csv')

# Let us consider the relationship between political instability and voter turnout
# One measure in V-Dem is the WB World Governance Indicators politcal stability index. Higher = more stable. This is variable e_wbgi_pve.
# Election turnout for a country and year is v2eltrnout.

# Summary Statistics
VDem %>%
    select(e_wbgi_pve, v2eltrnout) %>%
    data.frame() %>%
    stargazer(out="~/../repo/ECMA-31330-Project/Output/Summary_Statistics.tex", title="Summary Statistics", covariate.labels = c("WB Political Stability", "Election Turnout"))

# OLS Regression
OLS_instab_turnout <- lm(e_wbgi_pve ~ v2eltrnout, data = VDem)
stargazer(OLS_instab_turnout, se = starprep(OLS_instab_turnout, se_type = "stata"), title="Regression of WB Political Stability on Election Voter Turnout", covariate.labels = c("Election Turnout"), dep.var.labels = "WB Political Stability", dep.var.caption = "", align=TRUE, out="~/../repo/ECMA-31330-Project/Output/Instab_Turnout_Reg.tex", omit.stat=c("f", "ser", "adj.rsq"), notes = "Robust standard errors in parentheses.")
