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


```
python3 my_fenicsx_script.py
```
