# CSE 151B Competition

GPU type used was A30 and the total generation time is 1-1.5 minutes per question. 

We had no model weights.

To run the run_inference() just go to the command line and write out python run_inference.py. Note that data set that the model is run on should be called private.jsonl and be in the data folder. When it is done, it will create a .csv file called final_results.csv in the results folder. Before running the file, make sure to be using the .venv environment from the starter code. 

NOTE: In the submissions made to the Kaggle, we forgot to include the is_mcq column when submitting our .csvs. Only realized this mistake while making the run_inference.py file but that is fixed in it. We do not know if that will increase the accuracy or not since we could not generate another .csv in the time we had left. 
