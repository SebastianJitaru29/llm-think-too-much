This document describes how to run the L1 model with L1 Regressor and the BERT Regressor. Describing hidden_state_generation, regressor training and finally running the LLM model.


1. In data/processing/notebooks/data_extraction_jeroen.ipynb you can find the code for getting the gsm8k, olympiad and amc dataset and putting it in the right format
2. In data/processing/notebooks/hidden_state_generation.ipynb you can find the code for generating the hidden states for aime, math, gsm9k, olympiad and amc for both the L1 model and the BERT model.
3. In regressor/train.py you can find the script for training the regressor model.
(3 optional) For evaluation I build train_softmax.py. Instead of trying to predict all 20 bins, it predicts only the bin with the minimum required amount of tokens. This is mostly for evaluation purposes. This model is also able to predict if the question is impossible to answer.
4. In regressor/inference.py you find the script for running the just trained regressor model on the eval or test dataset. This gives an npy file with target tokens that can be used in 5. It might be worth here to play with the minium prob of a bin (p)
5. launch_regressor.py launches the LLM with the target number of tokens just calculated.
6. eval_regressor.ipynb contains some cells to evaluate the results obtained from 5