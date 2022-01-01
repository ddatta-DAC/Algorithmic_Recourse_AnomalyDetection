### First train the model

python3  AD_executor --dir us_import1

What does this do ?
1. Train the anomaly detection models. See the config --- which embedding dimensions are used to train the model.                        
2. Stores the p-th percentile values for the likelihood scores of training samples. This is needed for threshold based AD dtermination in our unsupervised model . By default : 2,5,10 th percentile values are calculated for each embedding dimension.
The saved dictionary is of the form  { <emb_dim>:  {2: <>, 5: <>, 10: <> }}                     
3. After the models have been trained ---
    import AD_Executor
    AD_Executor.setup_config(DIR)
    AD_Executor.read_models()
    
    >>> AD_Executor.score_new_sample(dataframe_row)


