# ME700-HW4

**How to run a fenicsx script on SCC**
Open a terminal on SCC and type the following commands to run a script that uses fenicsx.

```
module purge
module load fenicsx/0.5.1 
run_fenicsx.sh python3 my_fenicx_script.py
```
or alternatively, you can use a newer version of fenicsx(0.9.0). Loading this version is a liitle different because of its dependencies:  
```
module purge
module load openmpi/4.1.5_gnu-12.2.0
module load fenicsx/0.9.0
python3 my_fenicsx_script.py
```
Please note that you can check different fenicsx versions that are available on SCC using this command `module avail fenicsx`. Also, remember that certain notations in different versions of fenicsx can be different. 
