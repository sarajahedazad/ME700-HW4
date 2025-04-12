# ME700-HW4

**How to run a fenicsx script on SCC**
Open a terminal on SCC and type the following commands to run a script that uses fenicsx.

```
module purge
module load fenicsx/0.9.0
run_fenicsx.sh python3 my_fenicx_script.py
```
Please note that other versions of fenicsx are also available on SCC. You can check them using this command `module avail fenicsx`.
