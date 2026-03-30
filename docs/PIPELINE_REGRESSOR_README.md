This document describes how to run the L1 model with L1 Regressor and the BERT Regressor. Describing hidden_state_generation, regressor training and finally running the LLM model.


1. In data/processing/notebooks/data_extraction_jeroen.ipynb you can find the code for getting the gsm8k, olympiad and amc dataset and putting it in the right format
2. In data/processing/notebooks/hidden_state_generation.ipynb you can find the code for generating the hidden states for aime, math, gsm9k, olympiad and amc for both the L1 model and the BERT model.
3. In regressor/train.py you can find the script for training the regressor model.
(3 optional) For evaluation I build train_softmax.py. Instead of trying to predict all 20 bins, it predicts only the bin with the minimum required amount of tokens. This is mostly for evaluation purposes. This model is also able to predict if the question is impossible to answer.
4. In regressor/inference.py you find the script for running the just trained regressor model on the eval or test dataset. This gives an npy file with target tokens that can be used in 5. It might be worth here to play with the minium prob of a bin (p)
5. launch_regressor.py launches the LLM with the target number of tokens just calculated.
6. eval_regressor.ipynb contains some cells to evaluate the results obtained from 5


-----------------------------------------------------------------------------------------------------------
1. Run vllm_hack.py
2. In /venv/lib/python3.10/site-packages/vllm/platforms/interface.py line 212 change:
"return int(physical_device_id)" to "return physical_device_id"
3. Run python file with:
PYTHONMULTIPROCESSING_START_METHOD=spawn \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=MIG-03c7a8b7-dd9c-5bda-82fb-1cf2e9c3a34d \
python3 launch_regressor.py


source /scratch/s3799042/venvs/think-reduction/bin/activate

cd /home/s3799042/projects/llm-think-too-much


XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
PYTHONMULTIPROCESSING_START_METHOD=spawn \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=MIG-76de4d8d-dd90-510e-bc47-b5feb5c965b3 \
python3 launch_regressor.py



4: MIG-0fb6bae2-fd70-5e49-921b-9d8c9f43e593
7: MIG-5eaaabe9-dfae-59c1-bdbc-205f1e514960
8: MIG-76de4d8d-dd90-510e-bc47-b5feb5c965b3
9: MIG-03c7a8b7-dd9c-5bda-82fb-1cf2e9c3a34d