import click
import sys
import utils
from mpi4py import MPI
import time
import numpy as np
from dolfinx.io import XDMFFile
import basix
import basix.ufl
import dolfinx.fem as fem
import dolfinx.io as io
from dolfinx import default_scalar_type
from dolfinx.mesh import  locate_entities_boundary, meshtags
import dolfinx.plot as plot
import ufl
from dolfinx.mesh import CellType, create_rectangle,create_unit_square
from dolfinx.fem import (Expression, Function, functionspace,
                         assemble_scalar, dirichletbc,assemble_vector, form,Constant,locate_dofs_topological)

from dolfinx.fem.petsc import LinearProblem
import dolfinx.fem as fem
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector

import pyvista 

import os
import matplotlib.pyplot as plt
from classprojgrad import TAOProblem


pyvista.OFF_SCREEN = True
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numProc = comm.Get_size()

start_time = time.time()


# define The material property
def a(x):
    return (x**2)+ 1e-8
def aprime(x):
    return (2.0*x)


# define the double well  potential 
def W(x):
    return x**2 * (1.0-x)**2

def Wprime(x):

    return 2.0*x * (1.0- x)**2 -2.0 * x**2 * (1.0-x)




#%%%%%
@click.command()
@click.option(   '--output', '-o', type=click.STRING, default='rho.xdmf', help='result file' )
@click.option( '--eps', '-e', type=click.FLOAT, help="regularization parameter epsilon", default=0.015)
@click.option(  '--k1',  '-k' , type=click.FLOAT, help="scaling factor for the perimeter term", default =0.1)
@click.option ( '--nx', '-n', type=click.INT, help="the number of the element ", default= 350)    
@click.option(   '--k2', '-k2', type=click.FLOAT, help= "the weight on the penalty term", default= 0.8)        
@click.option ( '--mu', '-m', type= click.FLOAT, help=" step size", default= 9e-4)
@click.option(   '--lx', '-l', type=click.FLOAT, help="length on the x axis", default= 1.0   ) 
@click.option('--tao_max_it', '-tao_max_it', type=click.INT, help="Number of TAO iterations", default=100)
@click.option('--tao_max_funcs', '-tao_max_funcs', type =click.FLOAT, default = 10000, help = 'TAO maximum functions evaluations')
@click.option (  'ly', '-l', type=click.FLOAT, help= " length on y axis", default= 1.0   )
@click.option('--taoview', '-ta', is_flag=True, help="View convergence details")
@click.option('--tao_monitor', '-tao_monitor', is_flag= True, help= "TAO monitor"     )
@click.option('--tao_gatol', '-tao_gatol', type=float, default=1e-5, help="Stop if the norm of the gradient is less than this")
@click.option('--tao_gttol', '-tao_gttol', type=float, default=1e-5, help="Stop if the norm of the gradient is reduced by this factor")
@click.option('--tao_grtol', '-tao_grtol', type=float, default= 1e-5, help="Stop if the relative norm of gradient is less than this ")
@click.option( '--saveinterval' , '-s', type=click.INT, help="save interval", default=25)
@click.option(  '--theta','-th', type=click.FLOAT, help="volume constraint", default= 0.3)







  


def main(**options):
    
    #prefix, _ = os.path.splitext(options['output'])
    prefix,ext = os.path.splitext(options['output'])

    #mesh
    mesh=create_unit_square( comm, options['nx'], options['nx'], CellType.triangle)
    dx= ufl.Measure("dx", domain= mesh)

    V=fem.functionspace( mesh, ("Lagrange", 1))  # function space

    #define the the boundary
    def allBoundaries(x):
        return(
            #np.isclose(x[0], 0.0)|
            np.isclose(x[0], 1.0)|
            #np.isclose(x[1], 0.0)|
            np.isclose(x[1], 1.0)
            )

    allBoundariesFacets = locate_entities_boundary(mesh, mesh.topology.dim - 1, allBoundaries)
    allBoundariesDoF = fem.locate_dofs_topological(V, mesh.topology.dim - 1, allBoundariesFacets)
    bcAllBoundaries = fem.dirichletbc(0.0, allBoundariesDoF, V)

    #define the test and trial functions
    rho_n=fem.Function(V)
    rho=fem.Function(V)
    u= ufl.TrialFunction(V)
    v= ufl.TestFunction(V)
    g=fem.Constant( mesh, default_scalar_type(1.0))
    drho= ufl.TrialFunction(V)

    #cell_diameters= mesh.h(domain)
    #h_max=np.max(cell_diameters)
    #print(f"max_diameter: {h_max}")

    rho_lb=fem.Function(V, name="lower_bound")
    rho_ub=fem.Function(V, name="upper_bound")
    rho_lb.vector.set(-2.0)  # lower bound of rho
    rho_ub.vector.set(2.0)  # upper bound of rho

    # initialize rho
    rho_n.vector.set(0.3) 
    #rho_n.interpolate(lambda x: 0.3 + 0.3 * np.cos(4 * np.pi * x[0]) * np.cos(8 * np.pi * x[1]))

     # Synchronize rho 
    rho.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    rho.vector.setArray(rho_n.vector.getArray())   # copy rho_n to rho

    # Define and solve the PDE constraint
    A_constrn= a(rho)* ufl.inner( ufl.grad(u), ufl.grad(v))*ufl.dx
    L_constrn= g*v*ufl.dx
    PDEproblem= LinearProblem( A_constrn(u,v), L_constrn(v), bcs=[bcAllBoundaries ])
    uh=PDEproblem.solve()
    #print("uh max:", uh.x.array.max())
    



    #%%  the objective function 
    Compliance_ufl = g * uh * dx        # ds(1) 
    Perimeter_ufl  = options['k1'] * (W(rho) / options['eps'] + options['eps'] * ufl.inner(ufl.grad(rho), ufl.grad(rho)) ) * dx 
    J=Compliance_ufl + Perimeter_ufl

    #volume = fem.assemble_scalar(fem.form(1.0 * dx))   #the volume
    # compute the integral gradient of objective function
   
     #%% the gradient of the objective  function 
    #A_grad = drho*v*dx
    L_grad = options['k1'] * (Wprime(rho) / options['eps'] * v +  2. * options['eps']* ufl.inner(ufl.grad(rho), ufl.grad(v))) * dx    - aprime(rho)*ufl.inner(ufl.grad(uh), ufl.grad(uh))*v*dx

   

    J_form= form(J)
    gradJ_form = form(L_grad)


    #solve the problem 
    problem = TAOProblem(J_form, gradJ_form, rho,uh,PDEproblem,options, Perimeter_ufl, Compliance_ufl, options['output'],mesh)

    #Create TAO SOLVER 
    solver_tao = PETSc.TAO().create(MPI.COMM_WORLD)
    solver_tao.setType("lmvm") 
    solver_tao.setObjective(problem.ObjectiveFunction)
    solver_tao.setGradient(problem.GradObjectiveFunctionProjected)
    solver_tao.setFromOptions()
    solver_tao.setMonitor(problem.Monitor)
    solver_tao.setVariableBounds(rho_lb.vector, rho_ub.vector)  

         #set tolerance 
    solver_tao.setTolerances(
    gatol=options.get('taotol', 1e-4),
    grtol=options.get('taoGttol', 1e-4),
    gttol=options.get('taoGttol', 1e-4)
)
    
    


    x = rho.vector.copy()
    print("Initial x:", x.array)
    

    solver_tao.solve(x)

    reason_code = solver_tao.getConvergedReason()
 # get reason for convergence 
    reason_map = {
      0: "CONTINUE_ITERATING",         # Optimization not yet converged
      1: "TAO_CONVERGED_FATOL",            # Function absolute tolerance met
      2: "TAO_CONVERGED_FRTOL",            # Function relative tolerance met
      3: "TAO_CONVERGED_GATOL",            # Gradient absolute tolerance met
      4: "TAO_CONVERGED_GRTOL",            # Gradient relative tolerance met
      5: "TAO_CONVERGED_GTTOL",            # Gradient reduction tolerance met
      6: "TAO_CONVERGED_STEPTOL",          #step size smaller than tolerance
      7: "TAO_CONVERGED_MINF " ,            #function
      8:"TAO_CONVERGED_USER" ,             #the optimization has succeded 
      -1: "TAO_DIVERGED_NULL",             # Generic divergence (possibly undefined)
      -2: "TAO_DIVERGED_MAXITS",        #the maximum number of iterations allowed has been achieved  
      -3: "TAO_DIVERGED_MAXITS",           # Max number of iterations reached
      -4: " TAO_DIVERGED_NAN ",        # not a number appeared in the computations
      -5: " TAO_DIVERGED_MAXFCN ",      # the maximum number of function evaluations has been computed
      -6:"  TAO_DIVERGED_LS_FAILURE",   # a linesearch failed
      -7: " TAO_DIVERGED_TR_REDUCTION",  # trust region failure
      -8: "  TAO_DIVERGED_USER "         # optimization has failed
    
}
    reason_str = reason_map.get(reason_code, "UNKNOWN_REASON")

  

    if rank == 0:
        print(f"\n[TAO Convergence] Reason Code: {reason_code} ({reason_str})")



    
    
    rho.vector.array[:] = x.array  # Copy optimized values
    rho.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # Synchronize rho

     #save the plot 
    with XDMFFile(comm, f"{prefix}.xdmf", "w") as f:
        f.write_mesh(mesh)
        f.write_function(rho)


    solver_tao.destroy()

if __name__ == "__main__":
     main()











