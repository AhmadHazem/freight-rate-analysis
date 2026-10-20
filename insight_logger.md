# Documentation and logging the project
When I analyze any dataset I look for insights; those insights may raise thoughts or conclusions. Each insight has an identifier and it is linked to a thought

## Insights gained
- [INS1] There are unique 64 cities in train-val, and all of them exist in both pickup and delivery.
- [INS2] Interestingly, there are are 72 unique cities in the val, there are, extra 8 cities in the validation. Need to take that in consideration.
- [INS3] There 536 unique routes in train-test that is not in val, and there are 736 unique routes in val that are not in train-test.
- [INS4] Longitude and Latitude are consisent accross both train-test and val.
- [INS5] Distance is not a function of longitude and latitude
- [INS6] Visualizing posted_rate, weight, market_index and distance for the same random route, shows that cities, longitude, latitude are not needed. As despite having same route distance changes
- [INS7] There are 300 missing values in weight, 374 missing values in market_index in train-test. for validation, there are 165 in weight, and 249 missing values in market_index.
- [INS8] capping phenomen was observed on the weight feature
- [INS9] why are there negative weights ? maybe wrong input.
- [INS10] Distance is highly correlated with posted_rate. A very strong
feature
- [INS11] date plays a role in posted_rates, however it is not useful as a feature


## Thoughts to Check
- [INS1] -> [THOUGHT1]: We need as well to check the validation about the existence of same cities, or new cities were added to enusre no data shifts and consistency across train-test and validation
- [INS2 | INS2] -> [THOUGH2T]: the routes are not the same in both train-test and val. This difference discourges me from using cities as categorical features. I need to check one thing before dropping it entirely, which its impact on posted_rate. maybe despite having long distance posted rates can be low due to low demand.
- [THOUGHT3] Could Date play an important role ?
- [INS8] -> [THOUGH4] Check whether that capping will be highly impactful ?
- [THOUGHT5] What Possible imputation we could do for missing values ? we could impute based on the route
- [THOUGHT6] We could use date for the k fold splitter

## Conclusions
- [CONC1] Drop pickup,delivery,longitude,latitude
- [CONC2] Negative weight value seems like wrong input, use abs()
- [CONC3] Capping is about 2.5% of train-test and val as well, i think we can keep it as we may need to learn that capping, and the percentage is very small
- [CONC4] Drop date as feature but use if for k fold training