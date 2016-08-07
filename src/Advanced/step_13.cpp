/**
 * @file Modularity.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>
#include <list>
#include <sstream>

namespace Step13
{
//begin namespace Step13
using namespace dealii;

namespace Evaluation
{
//begin namespace Evaluation

/**
 * @class EvaluationBase
 */
template <int dim>
class EvaluationBase
{
public:
    virtual ~EvaluationBase ();

    void set_refinement_cycle (const size_t refinementCycle);

    virtual void operator () (const DoFHandler<dim> & dofHandler,
                              const Vector<double> & solution ) const = 0;
protected:
    size_t refinementCycle;
};

template <int dim>
EvaluationBase<dim>::~EvaluationBase()
{

}

/**
 * Set number of refinement cycle.
 * @param refinementCycle
 */
template <int dim>
void EvaluationBase<dim>::set_refinement_cycle(const size_t refinementCycle)
{
    this->refinementCycle = refinementCycle;
}

/**
 * @class PointValueEvaluation
 * With this class you can get value in the some points.
 */
template <int dim>
class PointValueEvaluation : public EvaluationBase<dim>
{
public:
    PointValueEvaluation(const Point <dim> &evaluationPoint,
                         TableHandler &resultsTable);

    virtual void operator()(const DoFHandler<dim> &dofHandler,
                            const Vector<double> &solution) const override;

    DeclException1 (ExcEvaluationPointNotFound,
                    Point<dim>,
                    << "The Evaluation point " << arg1
                        << " was not found among the vertices of the present grid");
private:
    const Point<dim> evaluationPoint;
    TableHandler & resultsTable;
};

/**
 * Constructor for PointValueEvaluation
 * @param evaluationPoint Point where we want to find solutions value.
 * @param resultsTable Table where we want to save soluton value.
 */
template <int dim>
PointValueEvaluation<dim>::PointValueEvaluation(const Point <dim> &evaluationPoint,
                                                TableHandler &resultsTable)
    :
    evaluationPoint(evaluationPoint),
    resultsTable(resultsTable)
{

}

/**
 * Functional call that look for point in dofHandler and draw value in to table
 * @param dofHandler
 * @param solution
 */
template <int dim>
void PointValueEvaluation<dim>::operator()(const DoFHandler<dim> &dofHandler,
                                           const Vector<double> &solution) const
{
    double pointValue = 1e20;

    bool evaluationPointFound = false;

    for (auto cell = dofHandler.begin_active();
         cell != dofHandler.end() && !evaluationPointFound; ++cell) {
        for (size_t v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
            if (cell->vertex(v) == evaluationPoint) {
                pointValue = solution(cell->vertex_dof_index(v, 0));
                evaluationPointFound = true;
                break;
            }
        }
    }

    AssertThrow(evaluationPointFound,
                ExcEvaluationPointNotFound(evaluationPoint));

    resultsTable.add_value("DoFs", dofHandler.n_dofs());
    resultsTable.add_value("u(x_0)", pointValue);
}

/**
 * @class SolutionOutput
 * Class to handle output.
 */
template <int dim>
class SolutionOutput : public EvaluationBase<dim>
{
public:
    SolutionOutput(const std::string &outputBaseName,
                   const DataOutBase::OutputFormat outputFormat);

    virtual void operator()(const DoFHandler<dim> &dofHandler,
                            const Vector<double> &solution) const override;

private:
    const std::string outputBaseName;
    const DataOutBase::OutputFormat outputFormat;
};

/**
 * Constructor
 * @param outputBaseName main part of the name
 * @param outputFormat file format in which file will write
 */
template <int dim>
SolutionOutput<dim>::SolutionOutput(const std::string &outputBaseName,
                                    const DataOutBase::OutputFormat outputFormat)
    :
    outputBaseName(outputBaseName),
    outputFormat(outputFormat)
{

}

/**
 * Write file
 * @param dofHandler
 * @param solution
 */
template <int dim>
void SolutionOutput<dim>::operator()(const DoFHandler<dim> &dofHandler,
                                     const Vector<double> &solution) const
{
    DataOut<dim> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.add_data_vector(solution, "solution");

    dataOut.build_patches();

    std::ostringstream filename;
    filename << outputBaseName << "-"
             << this->refinementCycle
             << dataOut.default_suffix(outputFormat)
             << std::ends;
    std::ofstream out (filename.str());

    dataOut.write(out, outputFormat);
}

//end namespace Evaluation
}

namespace LaplaceSolver
{
//begin namespace LaplaceSolver

/**
 * @class Base
 * Base class for diferent Laplace problem solvers.
 */
template <int dim>
class Base
{
public:
    Base (Triangulation<dim> & coarseGrid);
    virtual ~Base ();

    virtual void solve_problem () = 0;
    virtual void postprocess (const Evaluation::EvaluationBase<dim> &postprocessor) const = 0;
    virtual void refine_grid () = 0;
    virtual size_t n_dofs () const = 0;

protected:
    const SmartPointer<Triangulation<dim>> triangulation;
};

template <int dim>
Base<dim>::Base(Triangulation<dim> &coarseGrid)
    : triangulation (&coarseGrid)
{

}

template <int dim>
Base<dim>::~Base()
{

}


/**
 * @class Solver
 * Base
 */
template <int dim>
class Solver : public virtual Base<dim>
{
public:
    Solver (Triangulation<dim> & triangulation,
            const FiniteElement<dim> & fe,
            const Quadrature<dim> & quadrature,
            const Function<dim> & boundaryValues);

    virtual ~Solver ();

    virtual void solve_problem() override;
    virtual void postprocess (const Evaluation::EvaluationBase<dim> & postprocessor) const override;

    virtual size_t n_dofs () const override;

protected:
    const SmartPointer<const FiniteElement<dim>> fe;
    const SmartPointer<const Quadrature<dim>> quadrature;
    DoFHandler<dim> dofHandler;
    Vector<double> solution;
    const SmartPointer<const Function<dim>> boundaryValues;

    virtual void assemble_rhs (Vector<double> & rhs) const = 0;

private:
    struct LinearSystem
    {
        LinearSystem (const DoFHandler<dim> & dofHandler);

        void solve (Vector<double> & solution) const;

        ConstraintMatrix hangingNodeConstraints;
        SparsityPattern sparsityPattern;
        SparseMatrix<double> matrix;
        Vector<double> rhs;
    };

    struct AssemblyScratchData
    {
        AssemblyScratchData (const FiniteElement<dim> & fe,
                             const Quadrature<dim> & quadrature);
        AssemblyScratchData (const AssemblyScratchData & scratchData);

        FEValues<dim> feValues;
    };

    struct AssemblyCopyData
    {
        FullMatrix<double> cellMatrix;
        std::vector<types::global_dof_index> localDofIndices;
    };

    void assemble_linear_system (LinearSystem & linearSystem);
    void local_assemble_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                               AssemblyScratchData &scratchData,
                               AssemblyCopyData &copyData) const;

    void copy_local_to_global (const AssemblyCopyData & copyData,
                               LinearSystem & linearSystem) const;
};

/**
 * Constructor. Creates solver from mesh, finite element,
 * quadrature formula and function that determines boundary values.
 * @param triangulation mesh
 * @param fe finite element (e.g. Lagrange element)
 * @param quadrature quadrature formula (e.g. Gauss quadrature formula)
 * @param boundaryValues
 * @return
 */
template <int dim>
Solver<dim>::Solver(Triangulation<dim> &triangulation,
                    const FiniteElement<dim> &fe,
                    const Quadrature<dim> &quadrature,
                    const Function<dim> &boundaryValues)
    :
    Base<dim> (triangulation),
    fe (&fe),
    quadrature (&quadrature),
    dofHandler (triangulation),
    boundaryValues(&boundaryValues)
{

}

/**
 * Destructor. Clear DoFHandler.
 */
template <int dim>
Solver<dim>::~Solver()
{
    dofHandler.clear();
}

/**
 * Solves problem.
 */
template <int dim>
void Solver<dim>::solve_problem()
{
    dofHandler.distribute_dofs(*fe);
    solution.reinit(dofHandler.n_dofs());

    LinearSystem linearSystem (dofHandler);
    assemble_linear_system(linearSystem);
    linearSystem.solve(solution);
}

/**
 * Do postprocess (e.g. extract values from point, write output files)
 * @param postprocessor
 */
template <int dim>
void Solver<dim>::postprocess(const Evaluation::EvaluationBase<dim> &postprocessor) const
{
    postprocessor (dofHandler, solution);
}

/**
 * @return number of degrees of fredom of problem.
 */
template <int dim>
size_t Solver<dim>::n_dofs() const
{
    return dofHandler.n_dofs();
}

/**
 * Assemble linear system.
 * Uses parallel computing via multithreading.
 * @param linearSystem
 */
template <int dim>
void Solver<dim>::assemble_linear_system(LinearSystem &linearSystem)
{
    Threads::Task<> rhsTask = Threads::new_task(&Solver<dim>::assemble_rhs,
                                               *this,
                                               linearSystem.rhs);

    WorkStream::run (dofHandler.begin_active(),
                     dofHandler.end(),
                     std::bind(&Solver<dim>::local_assemble_matrix,
                               this,
                               std::placeholders::_1,
                               std::placeholders::_2,
                               std::placeholders::_3),
                     std::bind(&Solver<dim>::copy_local_to_global,
                               this,
                               std::placeholders::_1,
                               std::ref(linearSystem)),
                     AssemblyScratchData (*fe, *quadrature),
                     AssemblyCopyData());

    linearSystem.hangingNodeConstraints.condense(linearSystem.matrix);

    std::map<types::global_dof_index, double> boundaryValueMap;
    VectorTools::interpolate_boundary_values (dofHandler,
                                              0,
                                              *boundaryValues,
                                              boundaryValueMap);

    rhsTask.join();

    linearSystem.hangingNodeConstraints.condense(linearSystem.rhs);
    MatrixTools::apply_boundary_values (boundaryValueMap,
                                        linearSystem.matrix,
                                        solution,
                                        linearSystem.rhs);
}

/**
 * Constructor. Creates it from finite element and quadrature.
 * @param fe
 * @param quadrature
 */
template <int dim>
Solver<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe,
                                                      const Quadrature<dim> &quadrature)
    :
    feValues(fe, quadrature,
             update_gradients | update_JxW_values)
{

}

/**
 * Copy constructor.
 * @param scratchData
 */
template <int dim>
Solver<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratchData)
    :
    feValues (scratchData.feValues.get_fe(),
              scratchData.feValues.get_quadrature(),
              update_gradients | update_JxW_values)
{

}

/**
 * Assemble cell matrix.
 * @param cell iterator to current cell.
 * @param scratchData feValues wrapper object
 * @param copyData data where we want to write assembly matrix
 */
template <int dim>
void Solver<dim>::local_assemble_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                        AssemblyScratchData &scratchData,
                                        AssemblyCopyData &copyData) const
{
    const size_t dofsPerCell = fe->dofs_per_cell;
    const size_t nQPoints = quadrature->size();

    copyData.cellMatrix.reinit (dofsPerCell, dofsPerCell);

    copyData.localDofIndices.resize(dofsPerCell);

    scratchData.feValues.reinit(cell);

    for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
        for (size_t i = 0; i < dofsPerCell; ++i) {
            for (size_t j = 0; j < dofsPerCell; ++j) {
                copyData.cellMatrix(i, j) += (scratchData.feValues.shape_grad(i, qPoint) *
                                              scratchData.feValues.shape_grad(j, qPoint) *
                                              scratchData.feValues.JxW(qPoint));
            }
        }
    }

    cell->get_dof_indices(copyData.localDofIndices);
}

/**
 * Copy data from cell matrix to global matrix
 * @param copyData data to copy from
 * @param linearSystem system to copy data
 */
template <int dim>
void Solver<dim>::copy_local_to_global(const AssemblyCopyData &copyData,
                                       LinearSystem &linearSystem) const
{
    const size_t nLocalDofIndices = copyData.localDofIndices.size();

    for (size_t i = 0; i < nLocalDofIndices; ++i) {
        for (size_t j = 0; j < nLocalDofIndices; ++j) {
            linearSystem.matrix.add (copyData.localDofIndices[i],
                                     copyData.localDofIndices[j],
                                     copyData.cellMatrix(i, j));
        }
    }
}

/**
 * Constructor. Creates Linear system from dofHandler.
 * @param dofHandler
 */
template <int dim>
Solver<dim>::LinearSystem::LinearSystem(const DoFHandler<dim> &dofHandler)
{
    hangingNodeConstraints.clear();

    void (*mhnc_p) (const DoFHandler<dim> &, ConstraintMatrix &) =
        &DoFTools::make_hanging_node_constraints;

    Threads::Task<> sideTask = Threads::new_task(mhnc_p,
                                                 dofHandler,
                                                 hangingNodeConstraints);

    DynamicSparsityPattern dsp (dofHandler.n_dofs(), dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern (dofHandler, dsp);

    sideTask.join();

    hangingNodeConstraints.close();
    hangingNodeConstraints.condense(dsp);
    sparsityPattern.copy_from(dsp);

    matrix.reinit(sparsityPattern);
    rhs.reinit(dofHandler.n_dofs());
}

/**
 * Solve linear system and write solution in "solution" vector.
 * @param solution vector to output solution.
 */
template <int dim>
void Solver<dim>::LinearSystem::solve(Vector<double> &solution) const
{
    SolverControl solverControl (1000, 1e-12);
    SolverCG<> cg (solverControl);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(matrix, 1.2);

    cg.solve(matrix, solution, rhs, preconditioner);

    hangingNodeConstraints.distribute (solution);
}

/**
 * @class PrimalSolver
 * Implement solver with right hand side values.
 */
template <int dim>
class PrimalSolver : public Solver<dim>
{
public:
    PrimalSolver (Triangulation<dim> & triangulation,
                  const FiniteElement<dim> & fe,
                  const Quadrature<dim> & quadrature,
                  const Function<dim> & rhsFunction,
                  const Function<dim> & boundaryValues);

protected:
    const SmartPointer<const Function<dim>> rhsFunction;
    virtual void assemble_rhs(Vector<double> &rhs) const override;
};

/**
 * Constructor.
 * @param triangulation
 * @param fe
 * @param quadrature
 * @param rhsFunction
 * @param boundaryValues
 */
template <int dim>
PrimalSolver<dim>::PrimalSolver(Triangulation<dim> &triangulation,
                           const FiniteElement<dim> &fe,
                           const Quadrature<dim> &quadrature,
                           const Function<dim> &rhsFunction,
                           const Function<dim> &boundaryValues)
    :
    Base<dim> (triangulation),
    Solver<dim> (triangulation, fe,
                 quadrature, boundaryValues),
    rhsFunction (&rhsFunction)
{

}

/**
 * Assemble right hand side vector.
 * @param rhs vector in which we write assembled right hand side.
 */
template <int dim>
void PrimalSolver<dim>::assemble_rhs(Vector<double> &rhs) const
{
    FEValues<dim> feValues (*this->fe, *this->quadrature,
                            update_values | update_quadrature_points |
                            update_JxW_values);

    const size_t dofsPerCell = feValues.dofs_per_cell;
    const size_t nQPoints = this->quadrature->size();

    Vector<double> cellRhs(dofsPerCell);
    std::vector<double> rhsValues (nQPoints);
    std::vector<types::global_dof_index> localDofIndices (dofsPerCell);

    for (auto &&cell : this->dofHandler.active_cell_iterators()) {
        cellRhs = 0;
        feValues.reinit (cell);
        rhsFunction->value_list (feValues.get_quadrature_points(),
                                 rhsValues);

        for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
            for (size_t i = 0; i < dofsPerCell; ++i) {
                cellRhs (i) += (feValues.shape_value(i, qPoint) *
                                rhsValues[qPoint] *
                                feValues.JxW(qPoint));
            }
        }

        cell->get_dof_indices (localDofIndices);
        for (size_t i = 0; i < dofsPerCell; ++i) {
            rhs (localDofIndices[i]) += cellRhs (i);
        }
    }
}

/**
 * @class RefinementGlobal
 * Implement PrimalSolver with global refinement.
 */
template <int dim>
class RefinementGlobal : public PrimalSolver<dim>
{
public:
    RefinementGlobal(Triangulation<dim> &coarseGrid,
                     const FiniteElement<dim> &fe,
                     const Quadrature<dim> &quadrature,
                     const Function<dim> &rhsFunction,
                     const Function<dim> &boundaryValues);

    virtual void refine_grid () override;
};

/**
 * Constructor.
 * @param coarseGrid
 * @param fe
 * @param quadrature
 * @param rhsFunction
 * @param boundaryValues
 */
template <int dim>
RefinementGlobal<dim>::RefinementGlobal(Triangulation<dim> &coarseGrid,
                                        const FiniteElement<dim> &fe,
                                        const Quadrature<dim> &quadrature,
                                        const Function<dim> &rhsFunction,
                                        const Function<dim> &boundaryValues)
    :
    Base<dim>(coarseGrid),
    PrimalSolver<dim>(coarseGrid, fe,
                      quadrature, rhsFunction,
                      boundaryValues)
{

}

/**
 * Once do global refinement.
 */
template <int dim>
void RefinementGlobal<dim>::refine_grid()
{
    this->triangulation->refine_global(1);
}

/**
 * @class RefinementKelly
 * Implement PrimalSolver with local refinement uses Kelly refinement indicator.
 */
template <int dim>
class RefinementKelly : public PrimalSolver<dim>
{
public:
    RefinementKelly (Triangulation<dim> &coarseGrid,
                     const FiniteElement<dim> &fe,
                     const Quadrature<dim> &quadrature,
                     const Function<dim> &rhsFunction,
                     const Function<dim> &boundaryValues);

    virtual void refine_grid () override;
};

/**
 * Constructor.
 * @param coarseGrid
 * @param fe
 * @param quadrature
 * @param rhsFunction
 * @param boundaryValues
 * @return
 */
template <int dim>
RefinementKelly<dim>::RefinementKelly(Triangulation<dim> &coarseGrid,
                                      const FiniteElement<dim> &fe,
                                      const Quadrature<dim> &quadrature,
                                      const Function<dim> &rhsFunction,
                                      const Function<dim> &boundaryValues)
    :
    Base<dim>(coarseGrid),
    PrimalSolver<dim>(coarseGrid, fe, quadrature,
                      rhsFunction, boundaryValues)
{

}

/**
 * Refines mesh adaptivity using Kelly error estimator.
 */
template <int dim>
void RefinementKelly<dim>::refine_grid()
{
    Vector<float> estimatedErrorPerCell (this->triangulation->n_active_cells());
    KellyErrorEstimator<dim>::estimate (this->dofHandler,
                                        QGauss<dim - 1>(3),
                                        typename FunctionMap<dim>::type(),
                                        this->solution,
                                        estimatedErrorPerCell);
    GridRefinement::refine_and_coarsen_fixed_number (*this->triangulation,
                                                     estimatedErrorPerCell,
                                                     0.3, 0.03);
    this->triangulation->execute_coarsening_and_refinement ();
}

//end namespace LaplaceSolver
}

namespace ManufacturedSolution
{
//begin namespace ManufacturedSolution

template <int dim>
class Solution : public Function<dim>
{
public:
    Solution () : Function<dim>() {}

    virtual double value (const Point<dim> & p,
                        const size_t component = 0) const override;
};

/**
 * @param p
 * return .u(x,y)=exp(x+sin(10y+5x^2))
 */
template <int dim>
double Solution<dim>::value(const Point<dim> &p,
                            const size_t /*component*/) const
{
    double q = p(0);
    for (size_t i = 0; i < dim; ++i) {
        q += std::sin(10 * p(i) + 5 * p(0) * p(0));
    }
    const double exponential = std::exp(q);
    return exponential;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim> & p,
                          const size_t component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const size_t /*component*/) const
{
    double q = p(0);
    for (size_t i = 1; i < dim; ++i) {
        q += std::sin(10 * p(i) + 5 * p(0) * p(0));
    }
    const double u = std::exp(q);

    double t[3] = {1, 0, 0};
    for (size_t i = 1; i < dim; ++i) {
        t[0] += std::cos(10 * p(i) + 5 * p(0) * p(0)) * 10 * p(0);
        t[1] += 10 * std::cos(10 * p(i) + 5 * p(0) * p(0)) -
                100 * std::sin(10 * p(i) + 5 * p(0) * p(0));
        t[2] += 100 * std::cos(10 * p(i)+5 * p(0) * p(0)) *
                std::cos(10 * p(i)+5 * p(0) * p(0)) -
                100 * std::sin(10 * p(i)+5 * p(0) * p(0));
    }
    t[0] = t[0] * t[0];

    return - u * (t[0] + t[1] + t[2]);
}

//end namespace ManufacturedSolution
}

/**
 * Function that drives execution.
 * @param solver
 * @param postprocessorList
 */
template <int dim>
void run_simulation (LaplaceSolver::Base<dim> & solver,
                     const std::list<Evaluation::EvaluationBase<dim> *> &postprocessorList)
{
    std::cout << "Refinement cycle: ";
    for (size_t step = 0; true; ++step) {
        std::cout << step << "  " << std::flush;
        solver.solve_problem();

        for (auto postprocessor : postprocessorList) {
            postprocessor->set_refinement_cycle(step);
            solver.postprocess(*postprocessor);
        }

        if (solver.n_dofs() < 20000) {
            solver.refine_grid();
        }
        else {
            break;
        }
    }
    std::cout << std::endl;
}

/**
 * Create solver and run.
 * @param solveName
 */
template <int dim>
void solve_problem (const std::string & solverName)
{
    const std::string header = "Running tests with \"" + solverName +
                               "\" refinement criterion:";
    std::cout << header << std::endl
              << std::string (header.size(), '-') << std::endl;

    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (2);

    const FE_Q<dim> fe(1);
    const QGauss<dim> quadrature(4);
    const ManufacturedSolution::RightHandSide<dim> rhsFunction;
    const ManufacturedSolution::Solution<dim> boundaryValues;

    LaplaceSolver::Base<dim> * solver = nullptr;
    if (solverName == "global") {
        solver = new LaplaceSolver::RefinementGlobal<dim> (triangulation,
                                                           fe, quadrature,
                                                           rhsFunction,
                                                           boundaryValues);
    }
    else if (solverName == "kelly") {
        solver = new LaplaceSolver::RefinementKelly<dim> (triangulation,
                                                          fe, quadrature,
                                                          rhsFunction,
                                                          boundaryValues);
    }
    else {
        AssertThrow (false, ExcNotImplemented());
    }

    TableHandler resultsTable;
    Evaluation::PointValueEvaluation<dim> postprocessor1
        (Point<dim>(0.5, 0.5), resultsTable);
    Evaluation::SolutionOutput<dim> postprocessor2
        (std::string("solution-") + solverName, DataOutBase::vtu);
    std::list<Evaluation::EvaluationBase<dim> *> postprocessorList;

    postprocessorList.push_back(&postprocessor1);
    postprocessorList.push_back(&postprocessor2);

    run_simulation(*solver, postprocessorList);

    resultsTable.write_text(std::cout);
    delete solver;

    std::cout << std::endl;
}

//end namespace Step13
}


int main ()
{
    try {
        Step13::solve_problem<2> ("global");
        Step13::solve_problem<2> ("kelly");
    }
    catch (std::exception &exc)
    {
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
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    };
    return 0;
}