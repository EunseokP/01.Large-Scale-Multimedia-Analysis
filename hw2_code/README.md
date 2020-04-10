Full pipeline is given in the shell script **run.pipeline.sh**.

You can pass pass arguments to this bash script defining which one of the steps (preprocessing: **p**, feature representation: **f**, MAP scores: **m**, kaggle results: **k**, yaml filepath: **y**) you want to perform.

This helps you to avoid rewriting the bash script whenever there are intermediate steps that you don't want to repeat.
Here we also show you how to keep all your parameters in a **yaml file**. It helps to keep track of different parameter configurations that you may try. However, you do not have to keep your parameters in a yaml file. You can change this code as you want.

Here is an example of how to execute the script: 

    bash run.pipeline.sh -p true -f true -m true -y filepath
    

