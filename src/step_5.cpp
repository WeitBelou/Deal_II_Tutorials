/**
 *@file Work with external grids, optimizations, asserts,
 * changing viewport for eps output
*/
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//Read data from the disk
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
//C++ streams
#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;

template <int dim>
class Coefficient : public Function<dim>
{
public:
    Coefficient () : Function<dim>() {}

    virtual double value (const Point<dim> & p,
                          const unsigned int component = 0) const;
    virtual void value_list (const std::vector<Point<dim>>& points,
                             std::vector<double>& values,
                             const unsigned int component = 0) const;
};

template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int component) const
{
    if (p.square() < 0.25) {
        return 20;
    }
    else {
        return 1;
    }
}

template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim>> &points,
                                   std::vector<double> &values,
                                   const unsigned int component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    Assert (component == 0, ExcIndexRange(component, 0, 1));

    size_t nPoints = points.size();
    for (size_t i = 0; i < nPoints; ++i) {
        if (points[i].square() < 0.25) {
            values[i] = 20;
        }
        else {
            values[i] = 1;
        }
    }
}

template <int dim>
class ExtendedPoissonProblem
{
public:
         ExtendedPoissonProblem();
    void run (unsigned int nCycles);

private:
    void setupSystem();
    void assembleSystem();
    void solve ();
    void outputResults (const unsigned int cycle) const;


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
    ExtendedPoissonProblem<2> laplaceProblem2d;
    laplaceProblem2d.run(6);

    return 0;
}

template <int dim>
void ExtendedPoissonProblem<dim>::run (unsigned int nCycles)
{
    GridIn<dim> gridIn;
    gridIn.attach_triangulation(triangulation);

    std::ifstream inputFile("simpleRectGrid.ucd");

    Assert(dim == 2, ExcInternalError());

    gridIn.read_ucd(inputFile);

    static const FlatManifold<dim> boundary;
    triangulation.set_all_manifold_ids_on_boundary(0);
    triangulation.set_manifold(0, boundary);

    for (size_t cycle = 0; cycle < nCycles; ++cycle) {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle != 0) {
            triangulation.refine_global(1);
            std::cout << "    Number of active cells: "
                      << triangulation.n_active_cells()
                      << std::endl
                      << "    Total number of cells: "
                      << triangulation.n_cells()
                      << std::endl;
            setupSystem();
            assembleSystem();
            solve();
            outputResults(cycle);
        }
    }
}

template <int dim>
ExtendedPoissonProblem<dim>::ExtendedPoissonProblem () :
        fe(1), dofHandler(triangulation)
{

}

template <int dim>
void ExtendedPoissonProblem<dim>::setupSystem ()
{
    dofHandler.distribute_dofs(fe);

    std::cout << "Number of DoF: " << dofHandler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(dofHandler, dsp);
    sparsityPattern.copy_from(dsp);

    systemMatrix.reinit(sparsityPattern);

    solution.reinit(dofHandler.n_dofs());
    systemRhs.reinit(dofHandler.n_dofs());
}

template <int dim>
void ExtendedPoissonProblem<dim>::assembleSystem ()
{
    QGauss<dim> quadratureFormula(2);

    FEValues<dim> feValues(fe, quadratureFormula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

    const unsigned int dofsPerCell = fe.dofs_per_cell;
    const unsigned int nQPoints = quadratureFormula.size();

    FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
    Vector<double> cellRhs(dofsPerCell);

    std::vector<types::global_dof_index> localDofsIndices(dofsPerCell);

    const Coefficient<dim> coefficient;
    std::vector<double> coefficientValues(nQPoints);

    for (auto cell : dofHandler.active_cell_iterators()) {
        cellMatrix = 0;
        cellRhs = 0;

        feValues.reinit(cell);

        coefficient.value_list(feValues.get_quadrature_points(),
                               coefficientValues);
        for (size_t qIndex = 0; qIndex < nQPoints; ++qIndex) {
            for (size_t i = 0; i < dofsPerCell; ++i) {
                for (size_t j = 0; j < dofsPerCell; ++j) {
                    cellMatrix(i, j) += coefficientValues[qIndex] *
                                        feValues.shape_grad(i, qIndex) *
                                        feValues.shape_grad(j, qIndex) *
                                        feValues.JxW(qIndex);
                }

                cellRhs(i) += feValues.shape_value(i, qIndex) *
                              1.0 *
                              feValues.JxW(qIndex);
            }
        }

        cell->get_dof_indices (localDofsIndices);
        for (size_t i = 0; i < dofsPerCell; ++i) {
            for (size_t j = 0; j < dofsPerCell; ++j) {
                systemMatrix.add(localDofsIndices[i],
                                 localDofsIndices[j],
                                 cellMatrix(i, j));
            }
            systemRhs(localDofsIndices[i]) += cellRhs(i);
        }
    }
    std::map<types::global_dof_index, double> boundaryValues;
    VectorTools::interpolate_boundary_values(dofHandler, 0,
                                             ZeroFunction<dim>(),
                                             boundaryValues);
    MatrixTools::apply_boundary_values(boundaryValues,
                                       systemMatrix,
                                       solution,
                                       systemRhs);
}

template <int dim>
void ExtendedPoissonProblem<dim>::solve ()
{
    SolverControl solverControl(1000, 1e-12);
    SolverCG<> solver(solverControl);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(systemMatrix, 1.2);

    solver.solve(systemMatrix, solution, systemRhs, preconditioner);

    std::cout << "    " << solverControl.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}

template <int dim>
void ExtendedPoissonProblem<dim>::outputResults (const unsigned int cycle) const
{
    DataOut<dim> dataOut;

    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "magnitude");

    dataOut.build_patches();

    DataOutBase::EpsFlags epsFlags;

    epsFlags.z_scaling = 4;
    epsFlags.azimut_angle = 40;
    epsFlags.turn_angle = 10;

    dataOut.set_flags(epsFlags);

    std::ostringstream filename;
    filename << "solution-" << cycle << ".eps";

    std::ofstream out(filename.str());

    dataOut.write_eps(out);
}





