/**
 * @file Parallelization via MPI and PETSc.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace step17
{
    using namespace dealii;

    template <size_t dim>
    class ElasticProblem
    {
    public:
        ElasticProblem();
        ~ElasticProblem();
        void run();

    private:
        void setupSystem();
        void assembleSystem();
        size_t solve();
        void refineGrid();
        void outputResults(size_t cycle) const;

        MPI_Comm mpiCommunicator;

        const size_t nMpiProcesses;
        const size_t thisMpiProcess;

        ConditionalOStream pcout;

        Triangulation<dim> triangulation;
        DoFHandler<dim> dofHandler;

        FESystem<dim> fe;

        ConstraintMatrix hangingNodeConstraints;

        PETScWrappers::MPI::SparseMatrix systemMatrix;

        PETScWrappers::MPI::Vector solution;
        PETScWrappers::MPI::Vector systemRhs;
    };

    template<int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide ();

        virtual void vector_value (const Point<dim> &p,
                                   Vector<double> &values) const;

        virtual void vector_value_list (const std::vector<Point<dim>> &points,
                                        std::vector<Vector<double>> &valueList) const;
    };

    template<int dim>
    RightHandSide<dim>::RightHandSide () : Function<dim>()
    {

    }

    template<int dim>
    void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                           Vector<double> &values) const
    {
        Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
        Assert(dim >= 2, ExcNotImplemented());

        Point<dim> point1, point2;

        point1(0) = 0.5;
        point2(0) = -0.5;

        if (((p - point1).norm_square() < 0.04) ||
            ((p - point1).norm_square() < 0.04)) {
            values(1) = 1;
        }
        else {
            values(1) = 0;
        }

        if (p.norm_square() < 0.04) {
            values(1) = 1;
        }
        else {
            values(1) = 0;
        }
    }

    template<int dim>
    void RightHandSide<dim>::vector_value_list (
            const std::vector<Point<dim>> &points,
            std::vector<Vector<double>> &valueList) const
    {
        Assert(valueList.size() == points.size(),
               ExcDimensionMismatch(valueList.size(), points.size()));

        const size_t nPoints = points.size();

        for (size_t p = 0; p < nPoints; ++p) {
            RightHandSide<dim>::vector_value(points[p], valueList[p]);
        }
    }

    template <size_t dim>
    ElasticProblem<dim>::ElasticProblem() :
            mpiCommunicator(MPI_COMM_WORLD),
            nMpiProcesses (Utilities::MPI::n_mpi_processes(mpiCommunicator)),
            thisMpiProcess (Utilities::MPI::this_mpi_process(mpiCommunicator)),
            pcout (std::cout, (thisMpiProcess == 0)),
            dofHandler(triangulation),
            fe (FE_Q<dim>(1), dim)
    {

    }

    template <size_t dim>
    ElasticProblem<dim>::~ElasticProblem()
    {
        dofHandler.clear();
    }

    template <size_t dim>
    void ElasticProblem<dim>::run()
    {
        for (size_t cycle = 0; cycle < 10; ++cycle) {
            pcout << "Cycle: " << cycle << ':' << std::endl;

            if (cycle == 0) {
                GridGenerator::hyper_cube(triangulation, -1, 1);
                triangulation.refine_global(3);
            }
            else {
                refineGrid();
            }

            pcout << "    Number of active cells: "
                  << triangulation.n_active_cells()
                  << std::endl;

            setupSystem();

            pcout << "    Number of degrees of freedom: "
                  << dofHandler.n_dofs()
                  << " (by partition:";
            for (size_t p = 0; p < nMpiProcesses; ++p) {
                pcout << ((p == 0) ? (' ') : ('+'))
                      << (DoFTools::count_dofs_with_subdomain_association(dofHandler, p));
            }
            pcout << ')' << std::endl;

            assembleSystem();
            const size_t nIterations = solve();

            pcout << "    Solver converged in " << nIterations
                  << " iterations." << std::endl;

            outputResults(cycle);
        }
    }

    template <size_t dim>
    void ElasticProblem<dim>::setupSystem()
    {
        GridTools::partition_triangulation(nMpiProcesses, triangulation);

        dofHandler.distribute_dofs(fe);
        DoFRenumbering::subdomain_wise(dofHandler);

        const types::global_dof_index nLocalDofs =
                DoFTools::count_dofs_with_subdomain_association(dofHandler,
                                                                thisMpiProcess);

        systemMatrix.reinit(mpiCommunicator,
                            dofHandler.n_dofs(),
                            dofHandler.n_dofs(),
                            nLocalDofs,
                            nLocalDofs,
                            dofHandler.max_couplings_between_dofs());

        solution.reinit(mpiCommunicator, dofHandler.n_dofs(), nLocalDofs);
        systemRhs.reinit(mpiCommunicator, dofHandler.n_dofs(), nLocalDofs);

        hangingNodeConstraints.clear();
        DoFTools::make_hanging_node_constraints(dofHandler,
                                                hangingNodeConstraints);
        hangingNodeConstraints.close();
    }

    template <size_t dim>
    void ElasticProblem<dim>::assembleSystem()
    {
        QGauss<dim> quadratureFormula(2);
        FEValues<dim> feValues (fe, quadratureFormula,
                                update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

        const size_t dofsPerCell = feValues.dofs_per_cell;
        const size_t nQPoints = quadratureFormula.size();

        FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
        Vector<double> cellRhs(dofsPerCell);

        std::vector<types::global_dof_index> localDofIndices (dofsPerCell);

        std::vector<double> lambdaValues (nQPoints);
        std::vector<double> muValues (nQPoints);

        ConstantFunction<dim> lambda(1.0);
        ConstantFunction<dim> mu(1.0);

        RightHandSide<dim> rightHandSide;
        std::vector<Vector<double>> rhsValues(nQPoints,
                                              Vector<double>(dim));
        for (auto &&cell : dofHandler.active_cell_iterators()) {
            if (cell->subdomain_id() == thisMpiProcess) {
                cellMatrix = 0;
                cellRhs = 0;

                feValues.reinit(cell);

                lambda.value_list(feValues.get_quadrature_points(), lambdaValues);
                mu.value_list(feValues.get_quadrature_points(), muValues);

                for (size_t i = 0; i < dofsPerCell; ++i) {
                    const size_t componentI = fe.system_to_component_index(i).first;

                    for (size_t j = 0; j < dofsPerCell; ++j) {
                        const size_t componentJ = fe.system_to_component_index(j).first;

                        for (size_t qIndex = 0; qIndex < nQPoints; ++qIndex) {
                            cellMatrix(i, j) += (
                                                 (feValues.shape_grad(i, qIndex)[componentI] *
                                                  feValues.shape_grad(j, qIndex)[componentJ] *
                                                  lambdaValues[qIndex]
                                                 )
                                                 +
                                                 (
                                                  feValues.shape_grad(i, qIndex)[componentJ] *
                                                  feValues.shape_grad(j, qIndex)[componentI] *
                                                  muValues[qIndex]
                                                 )
                                                 +
                                                 ((componentI == componentJ) ?
                                                  (feValues.shape_grad(i, qIndex) *
                                                   feValues.shape_grad(j, qIndex) *
                                                   muValues[qIndex]) : 0)
                                                )
                                                * feValues.JxW(qIndex);
                        }
                    }

                }

                rightHandSide.vector_value_list(feValues.get_quadrature_points(),
                                                rhsValues);

                for (size_t i = 0; i < dofsPerCell; ++i) {
                    const size_t componentI =fe.system_to_component_index(i).first;

                    for (size_t qIndex = 0; qIndex < nQPoints; ++qIndex) {
                        cellRhs(i) += feValues.shape_value(i, qIndex) *
                                      rhsValues[qIndex](componentI) *
                                      feValues.JxW(qIndex);
                    }
                }
                cell->get_dof_indices (localDofIndices);
                hangingNodeConstraints.distribute_local_to_global(cellMatrix,
                                                                  cellRhs,
                                                                  localDofIndices,
                                                                  systemMatrix,
                                                                  systemRhs);
            }
        }

        systemMatrix.compress(VectorOperation::add);
        systemRhs.compress(VectorOperation::add);

        std::map<types::global_dof_index, double> boundaryValues;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 0,
                                                 ZeroFunction<dim>(dim),
                                                 boundaryValues);
        MatrixTools::apply_boundary_values(boundaryValues,
                                           systemMatrix,
                                           solution,
                                           systemRhs,
                                           false);
    }

    template <size_t dim>
    size_t ElasticProblem<dim>::solve()
    {
        SolverControl solverControl (solution.size(),
                                     1.0e-8 * systemRhs.l2_norm());
        PETScWrappers::SolverCG cg (solverControl,
                                    mpiCommunicator);

        PETScWrappers::PreconditionBlockJacobi preconditioner (systemMatrix);

        cg.solve(systemMatrix, solution, systemRhs, preconditioner);

        Vector<double> localizedSolution (solution);
        hangingNodeConstraints.distribute (localizedSolution);

        solution = localizedSolution;
        return solverControl.last_step();
    }

    template <size_t dim>
    void ElasticProblem<dim>::refineGrid()
    {
        const Vector<double> localizedSolution (solution);

        Vector<float> localErrorPerCell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate(dofHandler,
                                           QGauss<dim - 1>(2),
                                           typename FunctionMap<dim>::type(),
                                           localizedSolution,
                                           localErrorPerCell,
                                           ComponentMask(),
                                           0,
                                           MultithreadInfo::n_threads(),
                                           thisMpiProcess);

        const size_t nLocalCells =
                GridTools::count_cells_with_subdomain_association(triangulation,
                                                                  thisMpiProcess);
        PETScWrappers::MPI::Vector
                distributedAllErrors (mpiCommunicator,
                                      triangulation.n_active_cells(),
                                      nLocalCells);

        for (size_t i = 0; i < localErrorPerCell.size(); ++i) {
            if (localErrorPerCell(i) != 0) {
                distributedAllErrors(i) = localErrorPerCell(i);
            }
        }
        distributedAllErrors.compress(VectorOperation::insert);

        const Vector<float> localizedAllErrors (distributedAllErrors);

        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        localizedAllErrors,
                                                        0.3, 0.03);
        triangulation.execute_coarsening_and_refinement();
    }

    template <size_t dim>
    void ElasticProblem<dim>::outputResults(size_t cycle) const
    {
        const Vector<double> localizedSolution (solution);
        if (thisMpiProcess == 0) {
            std::ostringstream filename;
            filename << "solution-" << cycle << ".vtk";

            std::ofstream out (filename.str());

            DataOut<dim> dataOut;
            dataOut.attach_dof_handler(dofHandler);

            std::vector<std::string> solutionNames;
            switch (dim)
            {
                case 1:
                    solutionNames.push_back ("displacement");
                    break;
                case 2:
                    solutionNames.push_back("x_displacement");
                    solutionNames.push_back("y_displacement");
                    break;
                case 3:
                    solutionNames.push_back("x_displacement");
                    solutionNames.push_back("y_displacement");
                    solutionNames.push_back("z_displacement");
                    break;
                default:
                    Assert (false, ExcInternalError());
            }
            dataOut.add_data_vector (localizedSolution, solutionNames);

            std::vector<size_t> partitionInt (triangulation.n_active_cells ());
            GridTools::get_subdomain_association (triangulation, partitionInt);

            const Vector<double> partitioning (partitionInt.begin (),
                                               partitionInt.end ());
            dataOut.add_data_vector(partitioning, "partitioning");

            dataOut.build_patches();
            dataOut.write_vtk(out);
        }
    }


}

int main(int argc, char ** argv)
{
    try {
        using namespace dealii;
        using namespace step17;

        Utilities::MPI::MPI_InitFinalize mpiInitialization(argc, argv, 1);

        ElasticProblem<2> elasticProblem;
        elasticProblem.run();
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