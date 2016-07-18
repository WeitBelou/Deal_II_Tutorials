/**
 * @file Solve a simple version of Poisson's equation with zero boundary values,
 * but a nonzero right hand side
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

class Problem
{

public:
    Problem();
    void run();

private:
    void makeGrid();
    void setupSystem();
    void assemblySystem();
    void solve();
    void outputResults() const;

    Triangulation<2> triangulation;
    FE_Q<2> fe;
    DoFHandler<2> dofHandler;

    SparsityPattern sparsityPattern;
    SparseMatrix<double> systemMatrix;

    Vector<double> solution;

    //Right values
    Vector<double> systemRhs;
};

int main()
{
    deallog.depth_console(2);

    Problem laplaceProblem;
    laplaceProblem.run();
    return 0;
}

Problem::Problem () : fe(1), dofHandler(triangulation)
{

}

void Problem::run ()
{
    makeGrid();
    setupSystem();
    assemblySystem();
    solve();
    outputResults();
}

void Problem::makeGrid ()
{
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(5);
    std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void Problem::setupSystem ()
{
    dofHandler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: " << dofHandler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(dofHandler, dsp);
    sparsityPattern.copy_from(dsp);

    systemMatrix.reinit(sparsityPattern);

    solution.reinit(dofHandler.n_dofs());
    systemRhs.reinit(dofHandler.n_dofs());
}

void Problem::assemblySystem ()
{
    QGauss<2> quadratureFormula(2);
    FEValues<2> feValues(fe, quadratureFormula, update_values | update_gradients | update_JxW_values);
    const unsigned int doFsPerCell = fe.dofs_per_cell;
    const unsigned int numOfQPoints = quadratureFormula.size();

    FullMatrix<double> cellMatrix(doFsPerCell, doFsPerCell);
    Vector<double> cellRhs(doFsPerCell);

    std::vector<types::global_dof_index> localDoFIndices(doFsPerCell);

    for (auto cell : dofHandler.active_cell_iterators()) {
        feValues.reinit(cell);

        cellMatrix = 0;
        cellRhs = 0;

        for (size_t qIndex = 0; qIndex < numOfQPoints; ++qIndex) {

            for (size_t i = 0; i < doFsPerCell; ++i) {
                for (size_t j = 0; j < doFsPerCell; ++j) {
                    cellMatrix(i, j) += feValues.shape_grad(i, qIndex) *
                                        feValues.shape_grad(j, qIndex) *
                                        feValues.JxW(qIndex);
                }
            }

            for (size_t i = 0; i < doFsPerCell; ++i) {
                cellRhs(i) += feValues.shape_value(i, qIndex) *
                              1 *
                              feValues.JxW(i);
            }
        }

        cell->get_dof_indices(localDoFIndices);

        for (size_t i = 0; i < doFsPerCell; ++i) {
            for (size_t j = 0; j < doFsPerCell; ++j) {
                systemMatrix.add(localDoFIndices[i], localDoFIndices[j], cellMatrix(i, j));
            }
        }

        for (size_t i = 0; i < doFsPerCell; ++i) {
            systemRhs(localDoFIndices[i]) += cellRhs(i);
        }
    }

    std::map<types::global_dof_index, double> boundaryValues;
    VectorTools::interpolate_boundary_values(dofHandler, 0, ZeroFunction<2>(), boundaryValues);

    MatrixTools::apply_boundary_values(boundaryValues, systemMatrix, solution, systemRhs);
}

void Problem::solve ()
{
    SolverControl solverControl(1000, 1e-12);
    SolverCG<> solver(solverControl);
    solver.solve(systemMatrix, solution, systemRhs, PreconditionIdentity());
}

void Problem::outputResults () const
{
    DataOut<2> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "Solution");
    dataOut.build_patches();

    std::ofstream out("solution.vtu");
    dataOut.write_vtu(out);
}