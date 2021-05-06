# Defense_Spending_LASSO.R
# Run a lasso on the SIPRI defense spending data using glmnet

# Package Management
if ('pacman' %in% rownames(installed.packages()) == FALSE) {
  install.packages('pacman', repos='http://cran.us.r-project.org')
}
library(pacman)
p_load('tidyverse', 'glmnet', 'panelr', 'xtable')

# Detect whether this is a local run or on a computing cluster and set the file path appropriately
data_dir <- ifelse(dir.exists("~/../Box/ECMA-31330-Project"), "~/../Box/ECMA-31330-Project", "~/Box/ECMA-31330-Project")

# Load the data
sipri_for_LASSO <- read_csv(paste0(data_dir, '/SIPRI_for_LASSO.csv'), locale = readr::locale(encoding = "UTF-8"))

# One way of approaching this regression is to split the data into country subsamples

grouped_for_LASSO <- sipri_for_LASSO %>%
    group_by(Country)

# Save the group keys
country_keys <- grouped_for_LASSO %>%
    group_keys()

# Do the actual split into a list of dataframes
LASSO_split <- grouped_for_LASSO %>%
    group_split()

# Define a function to run the lasso on a country subsample's data
run_lasso <- function(data) {

    # Sadly, no missing data allowed
    data <- data %>%
        select(where(function(x) all(!is.na(x))))

    # Return NA if there's no variation in the dependent variable and hence no regression
    if (length(unique(data$Dep_Var_Spend)) == 1) {
        
        return(NA)
    
    } else {

        y <- data %>%
            select(Dep_Var_Spend) %>%
            data.matrix()

        X <- data %>%
            select(-Year, -Country, -Dep_Var_Spend) %>%
            data.matrix()

        # 10-fold CV with default settings should be fine
        # No penalty vector for now- penalize all coefficients
        # We may also want to try to estimate country or time specific parameters, but this will be difficult.
        cv_model = cv.glmnet(X, y, family="gaussian")

        chosen_lambda = cv_model['lambda.min'][[1]]
        lambdas = cv_model['lambda']
        cvmeans = cv_model['cvm']

        beta_lasso <- predict(cv_model, s = chosen_lambda, type="coefficients") %>%
            as.matrix() %>%
            as.data.frame() %>%
            as_tibble(rownames = "Coeff") %>%
            rename(Value = 2)

        return(beta_lasso)
    
    }

}

# Now we can make a dataframe of dataframes and apply this function to get some coefficients
country_lasso_results <- tibble(country_data = LASSO_split) %>%
    mutate(dep_var_country = country_keys, 
           beta_lasso = map(country_data, run_lasso)) %>%
    unnest(beta_lasso) %>%
    select(-country_data, -beta_lasso) %>%
    arrange(dep_var_country)

# Get the mean coefficient values across country LASSO specifications
mean_coeff_values <- country_lasso_results %>%
    group_by(Coeff) %>%
    summarize(mean_coeff = mean(Value))

write_csv(mean_coeff_values, "~/../repo/ECMA-31330-Project/Output/Regressions/LASSO_Country_Coeff_Values.csv")

coeff_table <- xtable(mean_coeff_values, type = "latex", tabular.environment="longtable")
print(coeff_table, include.rownames=TRUE, tabular.environment="longtable", floating=FALSE, file = "~/../repo/ECMA-31330-Project/Output/Regressions/LASSO_Country_Coeff_Values.tex")
