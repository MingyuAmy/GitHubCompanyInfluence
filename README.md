# GitHubCompanyInfluence
#### This is a project about predicting the influence of companies, using the performance on GitHub: h index.  
If the amount of api calls is limited, you can use your own token. (The token creation method is written in `common`.) If the code runs breaks, you can use a cache.  

`no2_hist.py` plots the company's social network, together with bars for each centrality metric. (Use common contributors as a basis for connecting edges.) 

`no1_plot_roc.py` is a prediction model using 11 features and XGBoost, and draw the roc curve.  
`no3_svm_f1.py` is a prediction model using 11 features and svm (takes a long time to run, please be patient)  
`no4_mlp_f1.py` is a prediction model using 11 features and mlp.    
Crawled data for all companies is stored in json format in `data_jsons.zip`  
All the output images are stored in `output`.  

This project crawled 486 companies' information, if you only want to use some of the companies data, you can modify in `data`.
