/**
 * @file Custom rhs and boundary values
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
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

using namespace dealii;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide() : Function<dim>() {}

    virtual double value (const Point<dim> & p, const unsigned int component = 0) const
    {
        double exponent = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            exponent += std::pow(p(i), 2);
        }
        return std::exp(exponent);
    }
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    BoundaryValues() : Function<dim>() {}

    virtual double value (const Point<dim> & p, const unsigned int component = 0) const
    {
        return p.square();
    }

};


template <int dim>
class TemplateProblem
{
public:
    TemplateProblem();
    void run();

private:
    void makeGrid();
    void setupSystem();
    void assemblySystem();
    void solve();
    void outputResults() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dofHandler;

    SparsityPattern sparsityPattern;
    SparseMatrix<double> systemMatrix;

    Vector<double> solution;
    Vector<double> systemRhs;
};

int main()
{
    deallog.depth_console(2);

    {
        TemplateProblem<2> laplaceProblem2d;
        laplaceProblem2d.run();
    }

    {
        TemplateProblem<3> laplaceProblem3d;
        laplaceProblem3d.run();
    }

    return 0;
}

template <int dim>
TemplateProblem<dim>::TemplateProblem () : fe(1), dofHandler(triangulation)
{

}

template <int dim>
void TemplateProblem<dim>::makeGrid ()
{
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(5);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
}

template <int dim>
void TemplateProblem<dim>::setupSystem ()
{
    dofHandler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: "
              << dofHandler.n_dofs()
              << std::endl;
    DynamicSparsityPattern dsp(dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(dofHandler, dsp);
    sparsityPattern.copy_from(dsp);

    systemMatrix.reinit(sparsityPattern);

    solution.reinit(dofHandler.n_dofs());
    systemRhs.reinit(dofHandler.n_dofs());
}

template <int dim>
void TemplateProblem<dim>::assemblySystem ()
{
    QGauss<dim> quadratureFormula(2);

    const RightHandSide<dim> rightHandSide;

    FEValues<dim> feValues(fe, quadratureFormula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

    const unsigned int dofsPerCell = fe.dofs_per_cell;
    const unsigned int nQPoints = quadratureFormula.size();

    FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
    Vector<double> cellRhs(dofsPerCell);

    std::vector<types::global_dof_index> localDofIndices(dofsPerCell);

    for (auto cell : dofHandler.active_cell_iterators()) {
        feValues.reinit(cell);
        cellMatrix = 0;
        cellRhs = 0;

        for (size_t qIndex = 0; qIndex < nQPoints; ++qIndex) {
            for (size_t i = 0; i < dofsPerCell; ++i) {
                for (size_t j = 0; j < dofsPerCell; ++j) {
                    cellMatrix(i, j) += feValues.shape_grad(i, qIndex) *
                                        feValues.shape_grad(j, qIndex) *
                                        feValues.JxW(qIndex);
                }
                cellRhs(i) += feValues.shape_value(i, qIndex) *
                              rightHandSide.value(feValues.quadrature_point(qIndex)) *
                              feValues.JxW(qIndex);
            }

            cell->get_dof_indices(localDofIndices);
            for (size_t i = 0; i < dofsPerCell; ++i) {
                for (size_t j = 0; j < dofsPerCell; ++j) {
                    systemMatrix.add(localDofIndices[i],
                                     localDofIndices[j],
                                     cellMatrix(i, j));
                }
                systemRhs(localDofIndices[i]);

            }
        }
    }

    std::map<types::global_dof_index, double> boundaryValues;
    VectorTools::interpolate_boundary_values(dofHandler, 0, BoundaryValues<dim>(), boundaryValues);

    MatrixTools::apply_boundary_values(boundaryValues, systemMatrix, solution, systemRhs);
}

template <int dim>
void TemplateProblem<dim>::solve ()
{
    SolverControl solverControl(1000, 1e-12);
    SolverCG<> solver(solverControl);
    solver.solve(systemMatrix, solution, systemRhs, PreconditionIdentity());

    std::cout << "    " << solverControl.last_step()
              << " CG iterations needed to obtain convergance."
              << std::endl;
}

template <int dim>
void TemplateProblem<dim>::outputResults () const
{
    DataOut<dim> dataOut;

    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "magnitude");

    dataOut.build_patches();

    std::ofstream out(dim == 2 ? "solution-2d.vtu" : "solution-3d.vtu");
    dataOut.write_vtu(out);
}

template <int dim>
void TemplateProblem<dim>::run ()
{
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

    makeGrid();
    setupSystem();
    assemblySystem();
    solve();
    outputResults();
}