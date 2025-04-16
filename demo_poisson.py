'''
This code is taken from this address:
https://github.com/FEniCS/dolfinx/blob/main/python/demo/demo_poisson.py
And the comments are added to it.

'''

# This module serves as a helper toolbox for finite element simulations and visualization.
# It integrates with libraries like DOLFINx, PETSc, and PyVista for solving and displaying solutions,
# while also ensuring the required dependencies are available.

import importlib.util  # Import module for dynamic imports and finding module specifications.

# =========================================================================================
# Explanation of importlib.util.find_spec and PETSc-related modules:
#
# ***importlib.util.find_spec("petsc4py")***
# What is a "spec"?
#   A spec (short for specification) is an object that informs Python how to import a module.
#   It contains details such as:
#     - The module's name
#     - Its location (file path)
#     - Its loader (the mechanism used to load it)
#     - If it's a package or a regular module, etc.
# This is represented as a ModuleSpec object from importlib.machinery.
# =========================================================================================

'''
petsc4py: a Python interface to PETSc.
PETSc (Portable, Extensible Toolkit for Scientific Computation) is a high-performance
library for solving large-scale linear and nonlinear systems, especially in parallel computing.
petsc4py enables the use of PETSc capabilities directly from Python.
'''

'''
Comparison between checks:
1. importlib.util.find_spec("petsc4py"):
   - Checks if the petsc4py module is installed.
   - If installed, Python will be able to import it; if not, a ModuleNotFoundError would occur.
   
2. if not dolfinx.has_petsc:
   - Checks if DOLFINx was compiled with PETSc support.
   - Even if petsc4py is installed, DOLFINx might be built without PETSc support.
   - This flag confirms whether PETSc is actually enabled within DOLFINx.
'''

# Check if the PETSc Python interface (petsc4py) is available.
if importlib.util.find_spec("petsc4py") is not None:  # If petsc4py is found...
    import dolfinx  # Import the DOLFINx finite element library.

    # Verify that DOLFINx has been compiled with PETSc support.
    if not dolfinx.has_petsc:  # If DOLFINx does not support PETSc...
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")  # Notify user.
        exit(0)  # Exit the program because PETSc support is mandatory.

    # Import ScalarType from PETSc to match the numeric type (float/complex) used by PETSc.
    from petsc4py.PETSc import ScalarType  # ScalarType ensures numeric consistency.
else:  # If petsc4py is not found...
    print("This demo requires petsc4py.")  # Notify user of the missing dependency.
    exit(0)  # Exit the program.

# =========================================================================================
# Import MPI from mpi4py to allow parallel computations using the Message Passing Interface.
# =========================================================================================
from mpi4py import MPI  # Enables parallel processing via MPI.

# Import numpy for numerical array operations.
import numpy as np  # Used for array manipulations and numerical functions.

# =========================================================================================
# Import UFL (Unified Form Language) to define variational forms for finite element discretizations.
# =========================================================================================
import ufl  # UFL is used to write mathematical formulations (weak forms) of PDEs.

# Import various modules from DOLFINx:
#   - fem: for creating function spaces and defining variational problems.
#   - io: for input/output operations (e.g., saving meshes and solutions).
#   - mesh: for generating and manipulating meshes.
#   - plot: for converting mesh data for visualization.
from dolfinx import fem, io, mesh, plot

# Import LinearProblem to define and solve the linear system derived from the weak formulation.
from dolfinx.fem.petsc import LinearProblem

# =========================================================================================
# Create a rectangular mesh.
# Define the domain by specifying two corners:
#   - Bottom-left: (0.0, 0.0)
#   - Top-right: (2.0, 1.0)
# The mesh is divided into cells:
#   - 32 subdivisions along the x-axis
#   - 16 subdivisions along the y-axis
# Cells are chosen to be triangles.
# =========================================================================================
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,                      # Use the global communicator for parallel runs.
    points=((0.0, 0.0), (2.0, 1.0)),           # Define the corners of the rectangle.
    n=(32, 16),                               # Number of cells in x and y directions.
    cell_type=mesh.CellType.triangle,         # Use triangles as the cell type.
)

# =========================================================================================
# Define a function space on the mesh using first-order (linear) Lagrange elements.
#
# ("Lagrange", 1) indicates:
#   - Lagrange elements with nodal basis functions.
#   - Polynomial degree 1 (linear approximation).
#
# Lagrange elements interpolate data, such that each basis function is one at its node and zero at others.
# =========================================================================================
V = fem.functionspace(msh, ("Lagrange", 1))  # Construct the finite element function space on the mesh.

# =========================================================================================
# Locate the boundary facets (edges in 2D) where the x-coordinate is either 0.0 or 2.0.
#
# For a 2D mesh, msh.topology.dim = 2; hence, boundary facets have dimension 1 (2-1).
# The lambda function returns True for points with x == 0.0 or x == 2.0.
# =========================================================================================
facets = mesh.locate_entities_boundary(
    msh,                                        # The mesh to analyze.
    dim=(msh.topology.dim - 1),                 # Identify edges (in 2D) or faces (in 3D).
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),  # Mark left and right boundaries.
)

# =========================================================================================
# Identify the degrees of freedom (DOFs) on the boundary facets for the function space V.
#
# fem.locate_dofs_topological returns indices of DOFs associated with topological entities (edges in this case).
# =========================================================================================
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)  # Get boundary DOFs.

# =========================================================================================
# Define a Dirichlet boundary condition for the problem.
#
# The boundary condition sets the solution u to zero on the boundary.
# ScalarType(0) ensures that the zero value is of the same numerical type (float or complex)
# that PETSc expects.
# =========================================================================================
bc = fem.dirichletbc(
    value=ScalarType(0),  # Value of zero matching PETSc's numeric type.
    dofs=dofs,            # Apply the condition to the degrees of freedom on the boundary.
    V=V                   # The function space where this condition is enforced.
)

# =========================================================================================
# Define the weak form of the Poisson problem.
#
# Weak Formulation:
#
#   Find u ∈ V (with u = 0 on the Dirichlet boundary) such that:
#
#       ∫_Ω ∇u ⋅ ∇v dx = ∫_Ω f v dx + ∫_(Γ_N) g v ds,   ∀ v ∈ V_0
#
# where:
#    - Ω is the computational domain.
#    - V is the finite element function space for the solution (with essential boundary conditions applied).
#    - V_0 is the space of test functions that vanish on the Dirichlet boundary.
#    - ∇ denotes the gradient.
#    - f is a source term defined on Ω.
#    - g is a boundary term (Neumann condition) defined on a part of the boundary.
#    - dx denotes integration over Ω.
#    - ds denotes integration over the boundary.
#
# In this problem:
#    - u is the trial function (unknown solution).
#    - v is the test function.
# =========================================================================================
u = ufl.TrialFunction(V)  # Define u as the trial function in the space V (the function we solve for).
v = ufl.TestFunction(V)   # Define v as the test function (used to weight the residual in the weak form).
x = ufl.SpatialCoordinate(msh)  # Get the spatial coordinates, allowing for spatially dependent expressions.

# =========================================================================================
# Define the source term f as a Gaussian-like function.
#
# f(x) = 10 * exp(- ((x[0] - 0.5)² + (x[1] - 0.5)²) / 0.02)
#
# This represents a localized forcing term within the domain.
# =========================================================================================
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)

# =========================================================================================
# Define the boundary term g as a sine function along the x-coordinate.
#
# g(x) = sin(5*x[0])
#
# This provides a Neumann-type contribution along the boundary.
# =========================================================================================
g = ufl.sin(5 * x[0])

# =========================================================================================
# Define the bilinear form a(·,·) corresponding to the left-hand side of the weak formulation.
#
# a(u, v) = ∫_Ω ∇u ⋅ ∇v dx
#
# This term represents the diffusion (or Laplacian) part of the Poisson equation.
# =========================================================================================
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # Integrate the inner product of gradients over the domain.

# =========================================================================================
# Define the linear form L(·) corresponding to the right-hand side of the weak formulation.
#
# L(v) = ∫_Ω f v dx + ∫_(boundary) g v ds
#
# The first term represents the source in the domain, while the second term accounts for the 
# boundary contribution.
# =========================================================================================
L = ufl.inner(f, v) * ufl.dx + ufl.inner(g, v) * ufl.ds  # Assemble the source and boundary contributions.

# =========================================================================================
# Set up and solve the linear problem using PETSc.
#
# The LinearProblem class automatically:
#    - Assembles the system of equations from the weak formulation.
#    - Applies the Dirichlet boundary conditions.
#    - Solves the resulting linear system using PETSc (with specified options).
#
# PETSc Options:
#   - ksp_type "preonly": Use only a preconditioner without an iterative solver.
#   - pc_type "lu": Use LU factorization for direct solving.
# =========================================================================================
problem = LinearProblem(
    a,                      # The bilinear form (stiffness matrix).
    L,                      # The linear form (right-hand side vector).
    bcs=[bc],               # List of boundary conditions.
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}  # Options for PETSc solver.
)
uh = problem.solve()  # Solve for uh, the finite element solution.

# =========================================================================================
# Write the mesh and the computed solution to an XDMF file.
#
# XDMF is a common file format for storing and visualizing scientific data, 
# and it can be easily used in visualization tools like ParaView.
# =========================================================================================
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)     # Save the mesh.
    file.write_function(uh)  # Save the computed solution.

# =========================================================================================
# Visualization using PyVista (if available).
#
# The following block attempts to:
#   1. Convert the DOLFINx mesh to a VTK-compatible format.
#   2. Create a PyVista UnstructuredGrid.
#   3. Attach the computed solution as point data.
#   4. Generate a plot showing the original and a warped (3D-extruded) version of the mesh.
#
# If PyVista is not installed, a message instructs the user to install it.
# =========================================================================================
try:
    import pyvista  # Import PyVista for advanced visualization.

    # Convert the finite element mesh into VTK format data.
    cells, types, x = plot.vtk_mesh(V)  # Get cell connectivity, cell types, and coordinates.
    grid = pyvista.UnstructuredGrid(cells, types, x)  # Create an unstructured grid for visualization.

    # Add the computed solution (uh) to the grid's point data.
    grid.point_data["u"] = uh.x.array.real  # Assign the real part of the solution to the grid.
    grid.set_active_scalars("u")  # Set the "u" field as the active scalar for plotting.

    # Create a PyVista Plotter window.
    plotter = pyvista.Plotter()  # Initialize the plotter.
    plotter.add_mesh(grid, show_edges=True)  # Add the grid with visible mesh edges for clarity.

    # Warp the mesh using the computed scalar field to provide a 3D visualization.
    warped = grid.warp_by_scalar()  # Create a warped (extruded) version of the grid.
    plotter.add_mesh(warped)  # Add the warped mesh to the plot.  

    # Now capture the screenshot.
    plotter.show(auto_close=False)
    plotter.screenshot("uh_poisson.png")
    plotter.close()

except ModuleNotFoundError:  # If PyVista is not installed...
    print("'pyvista' is required to visualise the solution.")  # Inform the user.
    print("To install pyvista with pip: 'python3 -m pip install pyvista'.")  # Provide installation guidance.


