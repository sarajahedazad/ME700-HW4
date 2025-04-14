# ME700-HW4
We start with creating and activating a conda environment. 
```
module load conda
conda create --name myfenicsxenv
conda activate myfenicsxenv
```
Or alternatively you can use miniconda:
```
module load miniconda
mamba create --name myfenicsxenv
mamba activate myfenicsxenv
```
Then we need to install
```
mamba install -c conda-forge fenics-dolfinx
mamba install -c conda-forge matplotlib
pip install pygmsh
python3 -m pip install pyvista
```
How to check our installations:
```
python -c "import dolfinx; print(dolfinx.__version__)"
python -c "import matplotlib; import pygmsh; print('matplotlib:', matplotlib.__version__, 'pygmsh:', pygmsh.__version__)"

```

And then you can 
```
python3 my_fenicsx_script.py
```
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
