# ME700-HW4
1. We start with creating and activating a conda environment. 
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
2. Then we need to install some dependencies
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
3. Run the `.py` file and look at the saved `.png` picture.
```
python3 my_fenicsx_script.py
```   

