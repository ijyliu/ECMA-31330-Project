# Defense_Spending_LASSO.R
# Run a lasso on the SIPRI defense spending data using glmnet

# Package Management
if ('pacman' %in% rownames(installed.packages()) == FALSE) {
  install.packages('pacman', repos='http://cran.us.r-project.org')
}
library(pacman)
p_load('tidyverse', 'glmnet')

# Detect whether this is a local run or on a computing cluster and set the file path appropriately
data_dir <- ifelse(dir.exists("~/../Box/ECMA-31330-Project"), "~/../Box/ECMA-31330-Project", "~/Box/ECMA-31330-Project")

# Load the data
sipri_for_LASSO <- read_csv(paste0('data_dir', '/SIPRI_for_LASSO.csv'))

# No penalty vector for now

y <- sipri_for_LASSO %>%
    select(Dep_Var_Spend) %>%
    data.matrix()

X <- patents_merged %>%
    select(-Year, -Country, -Dep_Var_Spend) %>%
    data.matrix()

cv_model = cv.glmnet(X, y, foldid = samples, penalty.factor = penalty_vector, family = "gaussian", intercept=FALSE)

chosen_lambda = cv_model['lambda.min'][[1]]
lambdas = cv_model['lambda']
cvmeans = cv_model['cvm']

chosen_beta_lasso <- predict(cv_model, s = chosen_lambda, type="coefficients") %>%
    as.matrix() %>%
    as.data.frame()
