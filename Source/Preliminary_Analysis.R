# Preliminary_Analysis.R
# Do some summary statistics on VDem

# Package Management
if ('pacman' %in% rownames(installed.packages()) == FALSE) {
  install.packages('pacman', repos='http://cran.us.r-project.org')
}
library(pacman)
p_load('tidyverse', 'stargazer', 'estimatr', 'plm', 'factoextra')

# Detect whether this is a local run or on a computing cluster and set the file path appropriately
data_dir <- ifelse(dir.exists("~/../Box/ECMA-31330-Project"), "~/../Box/ECMA-31330-Project", "~/Box/ECMA-31330-Project")

# Read in VDem data
# Note: if you want to read in a small subsample, set max_n = 1000 or some other value in read_csv
VDem <- read_csv(file.path(data_dir, '/V-Dem-CY-Full+Others-v11.1.csv'))
print("Loaded in V-Dem data.")

# General summary statistics for numeric variables
VDem %>%
  select_if(is.numeric) %>%
  data.frame() %>%
  stargazer(out="Output/Summary_Statistics_Numeric.tex", title="Summary Statistics for all Numeric Variables")

# Let us consider the relationship between political instability and voter turnout
# One measure in V-Dem is the WB World Governance Indicators politcal stability index. Higher = more stable. This is variable e_wbgi_pve.
# Election turnout for a country and year is v2eltrnout.

# Summary Statistics for the Regression Variables
VDem %>%
    select(e_wbgi_pve, v2eltrnout) %>%
    data.frame() %>%
    stargazer(out="Output/Summary_Statistics_for_Regression.tex", title="Summary Statistics for Regression Variables", covariate.labels = c("WB Political Stability", "Election Turnout"))

# OLS Regression
OLS_instab_turnout <- lm(e_wbgi_pve ~ v2eltrnout, data = VDem)
stargazer(OLS_instab_turnout, se = starprep(OLS_instab_turnout, se_type = "stata"), title="Regression of WB Political Stability on Election Voter Turnout", covariate.labels = c("Election Turnout"), dep.var.labels = "WB Political Stability", dep.var.caption = "", align=TRUE, out="Output/Instab_Turnout_Reg.tex", omit.stat=c("f", "ser", "adj.rsq"), notes = "Robust standard errors in parentheses.")

# Country-Year Fixed Effects Regression
FE_instab_turnout <- plm(e_wbgi_pve ~ v2eltrnout, data = VDem, index = c("country_name", "year"), model = "within")
# Get cluster SEs
summary(FE_instab_turnout, cluster="country_name") 
stargazer(FE_instab_turnout, se=list(coef(summary(FE_instab_turnout, cluster = c("country_name")))[, 2]), title="Fixed Effects Regression of WB Political Stability on Election Voter Turnout", covariate.labels = c("Election Turnout"), dep.var.labels = "WB Political Stability", dep.var.caption = "", align=TRUE, out="Output/FE_Instab_Turnout_Reg.tex", omit.stat=c("f", "ser", "adj.rsq"), notes = "Includes country and time fixed effects. Country-clustered standard errors in parentheses.")

# Principal Components Analysis
# Source: http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
# Let's do some exploratory work as to what are the major patterns in the data
# We need to pick some numeric variables along interesting dimensions
# For now I select all of the V2 or VDem version 2 variables that aren't strings
pca_results <- VDem %>%
    # Select VDem version 2 variables
    select(starts_with("v2")) %>%
    # Remove string columns
    select_if(is.numeric) %>%
    # Remove zero variance/constant columns
    select_if(function(v) var(v, na.rm=TRUE) != 0) %>%
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
