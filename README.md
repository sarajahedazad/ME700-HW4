The codes in this repository are written based on the tutorials in [this link](https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_poisson.py) and [this link](https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html).

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
3. Download and run the `.py` file and look at the saved `.png` picture. Be careful that you need to be in the same directory as the `.py` file.
```
python3 demo_poisson.py
```
---
**Running the Jupyter Notebook Tutorials**  
In case you want to run the jupyter notebooks in [tutorials folder](https://github.com/sarajahedazad/ME700-HW4/tree/main/tutorials), you might need to install Jupyter notebook in your conda environment:
```
pip install jupyter
```
```
cd tutorials/
```
```
jupyter notebook hw4_tut_partA.ipynb
```

