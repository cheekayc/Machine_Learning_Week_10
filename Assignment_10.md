Assignment 10
================
Chee Kay Cheong

``` r
knitr::opts_chunk$set(message = FALSE, warning = FALSE)

library(tidyverse)
library(caret)
library(rpart.plot)
```

# Part 1

``` r
# Load data file
load("./exposome.RData")

# Put all data frames into a list
df_list = list(covariates, exposome, phenotype) 

# Merge all data frames together
HW10 = df_list %>% 
  reduce(full_join, by = 'ID') %>% 
  select(-ID) %>% 
  mutate(hs_asthma = as_factor(hs_asthma))

# Check to see if there is missing values
miss.data = HW10[!complete.cases(HW10), ]
# no missing values in the dataset
```

The dataset `HW10` contains 1301 observations and 241 variables.

``` r
# Display rows and columns
dim(HW10)

# Summarize each variable in dataset
summary(HW10)
```

Descriptive measures:

- `h_trafload_preg_pow1over3` = Total traffic load of all roads in 100m
  buffer at pregnancy period (numeric)
  - Min : 0.3458  
  - 1st Quartile: 33.6542  
  - Median : 66.6101  
  - Mean : 75.5390  
  - 3rd Quartile: 113.0812  
  - Max : 294.2705
- `e3_asmokcigd_p_None` = maternal active tobacco smoke during pregnancy
  mean number of cigarette/day (numeric)
  - Min : 0.000  
  - 1st Quartile: 0.000  
  - Median : 0.000  
  - Mean : 0.494  
  - 3rd Quartile: 0.000  
  - Max : 15.238
- `h_pm10_ratio_preg_None` = pm10 value (extrapolated back in time using
  ratio method) during pregnancy (numeric)
  - Min : 8.066  
  - 1st Quartile: 17.535  
  - Median : 23.018  
  - Mean : 23.504  
  - 3rd Quartile: 27.677  
  - Max : 47.698
- `hs_pm10_wk_hs_h_None` = pm10 value (extrapolated back in time using
  ratio method) one week before hs test at home (numeric)
  - Min : 5.838  
  - 1st Quartile: 19.142  
  - Median : 24.891  
  - Mean : 26.409  
  - 3rd Quartile: 32.131  
  - Max : 211.297
- `h_lden_cat_preg_None` = Categorized lden (day, evening, night) at
  pregnancy period (numeric)
  - Min : 33.92  
  - 1st Quartile: 50.00  
  - Median : 58.63  
  - Mean : 57.47  
  - 3rd Quartile: 64.36  
  - Max : 77.40
- `hs_asthma` = Doctor diagnosed asthma (ever) (factor)
  - 0 (No) : 1159  
  - 1 (Yes) : 142

Examine correlation between variables:

``` r
# Create correlation matrix
round(cor(HW10[c('h_trafload_preg_pow1over3', 'e3_asmokcigd_p_None', 'h_pm10_ratio_preg_None', 'hs_pm10_wk_hs_h_None', 'h_lden_cat_preg_None')]), 2)
```

    ##                           h_trafload_preg_pow1over3 e3_asmokcigd_p_None
    ## h_trafload_preg_pow1over3                      1.00                0.03
    ## e3_asmokcigd_p_None                            0.03                1.00
    ## h_pm10_ratio_preg_None                         0.31                0.18
    ## hs_pm10_wk_hs_h_None                           0.08                0.13
    ## h_lden_cat_preg_None                           0.32                0.12
    ##                           h_pm10_ratio_preg_None hs_pm10_wk_hs_h_None
    ## h_trafload_preg_pow1over3                   0.31                 0.08
    ## e3_asmokcigd_p_None                         0.18                 0.13
    ## h_pm10_ratio_preg_None                      1.00                 0.49
    ## hs_pm10_wk_hs_h_None                        0.49                 1.00
    ## h_lden_cat_preg_None                        0.36                 0.11
    ##                           h_lden_cat_preg_None
    ## h_trafload_preg_pow1over3                 0.32
    ## e3_asmokcigd_p_None                       0.12
    ## h_pm10_ratio_preg_None                    0.36
    ## hs_pm10_wk_hs_h_None                      0.11
    ## h_lden_cat_preg_None                      1.00
