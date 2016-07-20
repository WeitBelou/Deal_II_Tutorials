/**
 * @file Adaptive local refinement. Higher order elements.
 *
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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

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
                                const unsigned int /*component*/) const
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
class AdaptiveLocalRefinment
{
public:
    AdaptiveLocalRefinment ();

    void run (unsigned int nCycles);

private:
    void setupSystem();
    void assembleSystem();
    void solve ();
    void refineGrid ();
    void outputResults (const unsigned int cycle) const;


    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dofHandler;

    ConstraintMatrix constraints;

    SparsityPattern sparsityPattern;
    SparseMatrix<double> systemMatrix;

    Vector<double> solution;
    Vector<double> systemRhs;
};

template <int dim>
AdaptiveLocalRefinment<dim>::AdaptiveLocalRefinment () :
    fe(2), dofHandler(triangulation)
{

}

template <int dim>
void AdaptiveLocalRefinment<dim>::setupSystem ()
{
    dofHandler.distribute_dofs(fe);

    solution.reinit(dofHandler.n_dofs());
    systemRhs.reinit(dofHandler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dofHandler, constraints);
    VectorTools::interpolate_boundary_values(dofHandler, 0,
                                             ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, false);

    sparsityPattern.copy_from(dsp);
    systemMatrix.reinit(sparsityPattern);
}

template <int dim>
void AdaptiveLocalRefinment<dim>::assembleSystem ()
{
    const QGauss<dim> quadratureFormula(3);

    FEValues<dim> feValues (fe, quadratureFormula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const size_t nQPoints = quadratureFormula.size();
    const size_t dofsPerCell = fe.dofs_per_cell;

    FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
    Vector<double> cellRhs(dofsPerCell);

    std::vector<types::global_dof_index> localDofIndices(dofsPerCell);

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

        cell->get_dof_indices(localDofIndices);
        constraints.distribute_local_to_global(cellMatrix,
                                               cellRhs,
                                               localDofIndices,
                                                systemMatrix,
                                                systemRhs);
    }
}

template <int dim>
void AdaptiveLocalRefinment<dim>::solve ()
{
    SolverControl solverControl(1000, 1e-12);
    SolverCG<> solver(solverControl);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(systemMatrix, 1.2);

    solver.solve(systemMatrix, solution, systemRhs, preconditioner);
    constraints.distribute(solution);
}

template <int dim>
void AdaptiveLocalRefinment<dim>::refineGrid ()
{
    Vector<float> estimatedErrorPerCell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dofHandler,
                                       QGauss<dim - 1>(3),
                                       typename FunctionMap<dim>::type(),
                                       solution,
                                       estimatedErrorPerCell);
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimatedErrorPerCell,
                                                    0.3, 0.03);
    triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void AdaptiveLocalRefinment<dim>::outputResults (const unsigned int cycle) const
{
    Assert(cycle < 10, ExcNotImplemented());

    std::string filename("grid-");
    filename += std::to_string(cycle);
    filename += ".eps";

    std::ofstream out(filename.c_str());

    GridOut gridOut;
    gridOut.write_eps(triangulation, out);
}

template <int dim>
void AdaptiveLocalRefinment<dim>::run (unsigned int nCycles)
{
    for (size_t cycle = 0; cycle < nCycles; ++cycle) {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0) {
            GridGenerator::hyper_ball(triangulation);

            static const SphericalManifold<dim> boundary;
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold(0, boundary);

            triangulation.refine_global(1);
        }
        else {
            refineGrid();
        }

        std::cout << "    Number of active cells:    "
                  << triangulation.n_active_cells()
                  << std::endl;

        setupSystem();

        std::cout << "    Number of degrees of freedom: "
                  << dofHandler.n_dofs()
                  << std::endl;

        assembleSystem();
        solve();
        outputResults(cycle);
    }

    DataOutBase::EpsFlags epsFlags;

    epsFlags.z_scaling = 4;

    DataOut<dim> dataOut;
    dataOut.set_flags(epsFlags);

    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "magnitude");
    dataOut.build_patches();

    std::ofstream output("final-solution.eps");
    dataOut.write_eps(output);
}

int main()
{
    try {
        AdaptiveLocalRefinment<2> laplaceProblem2d;
        laplaceProblem2d.run(6);
    }
    catch (std::exception & exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}