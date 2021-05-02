# Preliminary_Analysis.R
# Do some summary statistics on VDem

library(pacman)
p_load('tidyverse', 'stargazer', 'estimatr', 'plm', 'factoextra')

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

# Country-Year Fixed Effects Regression
FE_instab_turnout <- plm(e_wbgi_pve ~ v2eltrnout, data = VDem, index = c("country_name", "year"), model = "within")
# Get cluster SEs
summary(FE_instab_turnout, cluster="country_name") 
stargazer(FE_instab_turnout, se=list(coef(summary(FE_instab_turnout, cluster = c("country_name")))[, 2]), title="Fixed Effects Regression of WB Political Stability on Election Voter Turnout", covariate.labels = c("Election Turnout"), dep.var.labels = "WB Political Stability", dep.var.caption = "", align=TRUE, out="~/../repo/ECMA-31330-Project/Output/FE_Instab_Turnout_Reg.tex", omit.stat=c("f", "ser", "adj.rsq"), notes = "Includes country and time fixed effects. Country-clustered standard errors in parentheses.")

# Principal Components Analysis
# Source: http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
# Let's do some exploratory work as to what are the major patterns in the data
# We need to pick some numeric variables along interesting dimensions
# I chose the democracy indices (codebook section 2), suffrage indicators (codebook 3.1.1)
pca_results <- VDem %>%
    select(starts_with("v2x")) %>%
    prcomp(scale = TRUE)
# Visualize the eigenvectors/display the percentage of variance explained by each principal component
pdf("Output/Variance_Explained.pdf")
print(fviz_eig(pca_results))
dev.off()
# Visualize the directions
pdf("Output/Contributions_Directions.pdf")
print(fviz_pca_var(pca_results,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             ))
dev.off()
