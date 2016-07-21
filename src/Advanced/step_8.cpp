/**
 * @file Systems of PDE. Elasticity.
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
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

namespace step8
{
    using namespace dealii;

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


    template<int dim>
    class ElasticProblem
    {
    public:
        ElasticProblem ();

        ~ElasticProblem ();

        void run (const unsigned int nCycles);

    private:
        void setupSystem ();

        void assembleSystem ();

        void solve ();

        void refineGrid ();

        void outputResults (const unsigned int cycle) const;

        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dofHandler;
        FESystem<dim>        fe;
        ConstraintMatrix     hangingNodeConstraints;
        SparsityPattern      sparsityPattern;
        SparseMatrix<double> systemMatrix;
        Vector<double>       solution;
        Vector<double>       systemRhs;
    };

    template<int dim>
    ElasticProblem<dim>::ElasticProblem () :
            dofHandler(triangulation), fe(FE_Q<dim>(1), dim)
    {

    }

    template<int dim>
    ElasticProblem<dim>::~ElasticProblem ()
    {
        dofHandler.clear();
    }

    template<int dim>
    void ElasticProblem<dim>::run (const unsigned int nCycles)
    {
        for (size_t cycle = 0; cycle < nCycles; ++cycle) {
            std::cout << "Cycle " << cycle << ':' << std::endl;

            if (cycle == 0) {
                GridGenerator::hyper_cube(triangulation, -1, 1);
                triangulation.refine_global(2);
            }
            else {
                refineGrid();
            }

            std::cout << "   Number of active cells:       "
                      << triangulation.n_active_cells()
                      << std::endl;
            setupSystem();

            std::cout << "   Number of degrees of freedom: "
                      << dofHandler.n_dofs()
                      << std::endl;

            assembleSystem();
            solve();
            outputResults(cycle);
        }
    }

    template<int dim>
    void ElasticProblem<dim>::setupSystem ()
    {
        dofHandler.distribute_dofs(fe);

        hangingNodeConstraints.clear();
        DoFTools::make_hanging_node_constraints(dofHandler,
                                                hangingNodeConstraints);
        hangingNodeConstraints.close();

        DynamicSparsityPattern dsp(dofHandler.n_dofs(), dofHandler.n_dofs());
        DoFTools::make_sparsity_pattern(dofHandler, dsp,
                                        hangingNodeConstraints, true);
        sparsityPattern.copy_from(dsp);

        systemMatrix.reinit(sparsityPattern);

        solution.reinit(dofHandler.n_dofs());
        systemRhs.reinit(dofHandler.n_dofs());
    }

    template<int dim>
    void ElasticProblem<dim>::assembleSystem ()
    {
        QGauss<dim> quadratureFormula(2);

        FEValues<dim> feValues(fe, quadratureFormula,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

        const size_t dofsPerCell = fe.dofs_per_cell;
        const size_t nQPoints    = quadratureFormula.size();

        FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
        Vector<double>     cellRhs(dofsPerCell);

        std::vector<types::global_dof_index> localDofIndices(dofsPerCell);

        std::vector<double> lambdaValues(nQPoints);
        std::vector<double> muValues(nQPoints);

        ConstantFunction<dim> lambda(1.0), mu(1.0);

        RightHandSide<dim>          rightHandSide;
        std::vector<Vector<double>> rhsValues(nQPoints, Vector<double>(dim));

        for (auto cell : dofHandler.active_cell_iterators()) {
            cellMatrix = 0;
            cellRhs    = 0;

            feValues.reinit(cell);

            lambda.value_list(feValues.get_quadrature_points(), lambdaValues);
            mu.value_list(feValues.get_quadrature_points(), muValues);

            rightHandSide.vector_value_list(feValues.get_quadrature_points(),
                                            rhsValues);

            for (size_t i = 0; i < dofsPerCell; ++i) {
                const size_t componentI = fe.system_to_component_index(i).first;

                for (size_t j = 0; j < dofsPerCell; ++j) {
                    const size_t componentJ = fe.system_to_component_index(
                            j).first;

                    for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
                        cellMatrix(i, j) +=
                            (
                                (feValues.shape_grad(i, qPoint)[componentI] *
                                 feValues.shape_grad(j, qPoint)[componentJ] *
                                 lambdaValues[qPoint])
                                +
                                (feValues.shape_grad(i, qPoint)[componentJ] *
                                 feValues.shape_grad(j, qPoint)[componentI] *
                                 muValues[qPoint])
                                +
                                ((componentI == componentJ)
                                 ?
                                 (feValues.shape_grad(i, qPoint) *
                                  feValues.shape_grad(j, qPoint) *
                                  muValues[qPoint])
                                 :
                                 (0)
                                )
                            ) * feValues.JxW(qPoint);
                    }
                }
            }

            for (size_t i = 0; i < dofsPerCell; ++i) {
                const size_t componentI = fe.system_to_component_index(i).first;

                for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
                    cellRhs(i) += feValues.shape_value(i, qPoint) *
                                  rhsValues[qPoint](componentI) *
                                  feValues.JxW(qPoint);
                }
            }

            cell->get_dof_indices(localDofIndices);
            for (size_t i = 0; i < dofsPerCell; ++i) {
                for (size_t j = 0; j < dofsPerCell; ++j) {
                    systemMatrix.add(localDofIndices[i], localDofIndices[j],
                                     cellMatrix(i, j));
                }

                systemRhs(localDofIndices[i]) += cellRhs(i);
            }
        }

        hangingNodeConstraints.condense(systemMatrix);
        hangingNodeConstraints.condense(systemRhs);

        std::map<types::global_dof_index, double> boundaryValues;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 0,
                                                 ZeroFunction<dim>(dim),
                                                 boundaryValues);
        MatrixTools::apply_boundary_values(boundaryValues,
                                           systemMatrix,
                                           solution,
                                           systemRhs);
    }

    template<int dim>
    void ElasticProblem<dim>::solve ()
    {
        SolverControl solverControl(1000, 1e-12);
        SolverCG<>    solver(solverControl);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(systemMatrix, 1.2);

        solver.solve(systemMatrix, solution, systemRhs, preconditioner);
        hangingNodeConstraints.distribute(solution);
    }

    template<int dim>
    void ElasticProblem<dim>::refineGrid ()
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

    template<int dim>
    void ElasticProblem<dim>::outputResults (const unsigned int cycle) const
    {
        std::string filename("solution-");

        Assert(cycle < 10, ExcInternalError());

        filename += std::to_string(cycle);
        filename += ".vtk";

        std::ofstream out(filename);

        DataOut<dim> dataOut;
        dataOut.attach_dof_handler(dofHandler);

        std::vector<std::string> solutionNames;
        switch (dim) {
            case 1:
                solutionNames.push_back("displacement");
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
            default: Assert(false, ExcNotImplemented());
        }

        dataOut.add_data_vector(solution, solutionNames);
        dataOut.build_patches();
        dataOut.write_vtk(out);
    }
}

int main ()
{
    try
    {
        step8::ElasticProblem<2> elasticProblem2d;
        elasticProblem2d.run(6);
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
    }
return 0;
}