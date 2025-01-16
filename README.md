# GenBen: A Benchmark for LLM-Aided Design
 GenBen (Generative Hardware Benchmark)is a new benchmark for evaluating large language models (LLMs) in hardware design.We have provided some of the problems in data.zip. We have provided some problems in the data.zip file, along with the corresponding executable code.

## Notices

The project has completed the LLM tests of the GPT and Claude series. Currently, we are conducting tests on other mainstream LLMs such as Gemini, QWen, and DeepSeek, aiming to adjust and optimize the compatibility and universality of the tests.

All the code, test cases, and our current test reports will be released after the Chinese New Year holiday, that is, no later than February 21st. Please stay tuned.


## Install
We closely follow guidance from VerilogEval(https://github.com/joshual-xlnx/verilog-eval) and notebook.ipynb 
## Usage  
### In the first step, You can decompress the data.zip in the folder using the following designations.
```
unzip data.zip
```
### In the second step, You can run the following command to extract the relevant topic description from the data folder in the first step and generate multiple json files according to different modalities:
```
python get_data.py
```
For example:
 We used the above directive to obtain some new descriptions 'descriptions_test_golden_top_debug_code.jsonl';'descriptions_test_golden_top_code_mm.jsonl';'descriptions_test_golden_top_code_all.jsonl';'descriptions_test_golden_top_code_text.jsonl' and save them in save_data file.
### In the third step, you can use DS_get_ask.py to obtain the problem description after adding disturbances; and use get_new_ds.py to obtain the JSON file of the new disturbed problem description.
```
python DS_get_ask.py
python get_new_ds.py
```
### In the last step, please be sure to modify the requests_filepath in genben.py to the newly generated JSON file from the previous step. After completing this, run the following code:
```python genben.py --mode all --model gpt4```
#### Command to Run the Evaluation with Specified Parameters

##### Understanding the Command Parameters

- **--mode**: This parameter controls the type of tasks input into the LLMs. There are three available options:
  - **all**: Enables the input of all task types.
  - **mm**: Allows for multi-modal tasks.
  - **text**: Restricts the input to text-based tasks only.

- **--model**: This parameter specifies the model of the LLMs. Adjust this parameter according to the specific API of the LLMs you are using.


After running, the results will be output from various large models that do not pass the problem, in addition to the score of the knowledge understanding and transfer questions and save them in the 'choice_pass.txt', and the pass@5 of the debug problems and save the results in the 'debug_pass.txt'. For code generation problems, the PPA, funcitonal, synatx synthesizbility results will be generated, and the PPA results will be saved in the result_pass.jsonl file, and the funcitonal and synatx results will be saved in the eval_{args. mode}_{args.model}_code.jsonl_results.jsonl'. The results of Synthesizbility will be stored in ''eval_{args.} mode}_{args.model}_code.jsonl_results_yosys_result.jsonl’。


