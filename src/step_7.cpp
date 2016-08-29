/**
  *@file Evaluating error. Non-homohenious boundary conditions.
  */

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/convergence_table.h>

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
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>

#include <typeinfo>
#include <fstream>
#include <iostream>

namespace step7
{
    using namespace dealii;

    /**
     * @class SolutionBase template class for solutions.
     */
    template <int dim>
    class SolutionBase
    {
    protected:
        static const size_t nSourceCenters = 3;
        static const Point<dim> sourceCenters[nSourceCenters];
        static const double width;
    };

    template <>
    const Point<1> SolutionBase<1>::sourceCenters[SolutionBase<1>::nSourceCenters]
        = { Point<1>(-1.0 / 3.0),
            Point<1>(0.0),
            Point<1>(1.0 / 3.0)
          };

    template <>
    const Point<2> SolutionBase<2>::sourceCenters[SolutionBase<2>::nSourceCenters]
        = { Point<2>(-0.5, +0.5),
            Point<2>(-0.5, -0.5),
            Point<2>(+0.5, -0.5)
          };


    template <int dim>
    const double SolutionBase<dim>::width = 1.0 / 8.0;

    /**
     * @class Solution provides exact solution.
     */
    template <int dim>
    class Solution : public Function<dim>, protected SolutionBase<dim>
    {
    public:
        Solution();

        virtual double value (const Point<dim> &p,
                              const unsigned int component = 0) const;

        virtual Tensor<1, dim> gradient(const Point<dim> &p,
                                        const unsigned int component = 0) const;
    };

    template <int dim>
    Solution<dim>::Solution() : Function<dim>()
    {

    }

    template <int dim>
    double Solution<dim>::value(const Point<dim> & p,
                           const unsigned int /*component*/) const
    {
        double returnValue = 0.0;
        const double widthSquare = this->width * this->width;

        for (size_t i = 0; i < this->nSourceCenters; ++i) {
            const Tensor<1, dim> xMinusXi = p - this->sourceCenters[i];
            returnValue += std::exp(-xMinusXi.norm_square() /
                                    widthSquare);
        }
        return returnValue;
    }

    template <int  dim>
    Tensor<1, dim> Solution<dim>::gradient(const Point<dim> & p,
                                      const unsigned int /*component*/) const
    {
        Tensor<1, dim> returnValue;
        const double widthSquare = this->width * this->width;

        for (size_t i = 0; i < this->nSourceCenters; ++i) {
            const Tensor<1, dim> xMinusXi = p - this->sourceCenters[i];

            returnValue += (-2 / widthSquare *
                            std::exp(-xMinusXi.norm_square() / widthSquare) *
                            xMinusXi
                            );
        }
        return returnValue;
    }

    /**
     * @brief The RightHandSide class
     */
    template <int dim>
    class RightHandSide : public Function<dim>, protected SolutionBase<dim>
    {
    public:
        RightHandSide();

        virtual double value (const Point<dim> &p,
                              const unsigned int component) const;
    };

    template <int dim>
    RightHandSide<dim>::RightHandSide() : Function<dim>()
    {

    }

    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> & p,
                                     const unsigned int /*component*/) const
    {
        double returnValue = 0;
        const double widthSquare = this->width * this->width;

        for (size_t i = 0; i < this->nSourceCenters; ++i) {
            const Tensor<1, dim> xMinusXi = p - this->sourceCenters[i];

            returnValue += ((2 * dim - 4 * xMinusXi.norm_square() /
                            widthSquare) / widthSquare *
                            std::exp(-xMinusXi.norm_square() / widthSquare)
                            );

            returnValue += std::exp(-xMinusXi.norm_square() / widthSquare);
        }

        return returnValue;
    }

    /**
     * @brief The HelmholzProblem class
     */
    template <int dim>
    class HelmholtzProblem
    {
    public:
        enum RefinementMode
        {
            globalRefinement, adaptiveRefinement
        };

        HelmholtzProblem(const FiniteElement<dim> & fe,
                        const RefinementMode refinementMode);
        ~HelmholtzProblem();

        void run();

    private:
        HelmholtzProblem() {}

        void setupSystem();
        void assembleSystem();
        void solve();
        void refineGrid();
        void processSolution(const size_t cycle);

        Triangulation<dim> triangulation;
        DoFHandler<dim> dofHandler;

        SmartPointer<const FiniteElement<dim>> fe;

        ConstraintMatrix hangingNodeConstraints;

        SparsityPattern sparsityPattern;
        SparseMatrix<double> systemMatrix;

        Vector<double> solution;
        Vector<double> systemRhs;

        const RefinementMode refinementMode;

        ConvergenceTable convergenceTable;
    };

    template <int dim>
    HelmholtzProblem<dim>::HelmholtzProblem(const FiniteElement<dim> & fe,
                                     const HelmholtzProblem::RefinementMode refinementMode) :
        dofHandler(triangulation), fe(&fe), refinementMode(refinementMode)
    {

    }

    template <int dim>
    HelmholtzProblem<dim>::~HelmholtzProblem()
    {
        dofHandler.clear();
    }

    template <int dim>
    void HelmholtzProblem<dim>::setupSystem()
    {
        dofHandler.distribute_dofs(*fe);
        DoFRenumbering::Cuthill_McKee(dofHandler);

        hangingNodeConstraints.clear();
        DoFTools::make_hanging_node_constraints(dofHandler,
                                                hangingNodeConstraints);
        hangingNodeConstraints.close();

        DynamicSparsityPattern dsp (dofHandler.n_dofs(), dofHandler.n_dofs());
        DoFTools::make_sparsity_pattern (dofHandler, dsp);
        hangingNodeConstraints.condense(dsp);
        sparsityPattern.copy_from(dsp);

        systemMatrix.reinit(sparsityPattern);

        solution.reinit(dofHandler.n_dofs());
        systemRhs.reinit(dofHandler.n_dofs());
    }

    template <int dim>
    void HelmholtzProblem<dim>::assembleSystem()
    {
        QGauss<dim> quadratureFormula(3);
        QGauss<dim - 1> faceQuadratureFormula(3);

        const size_t nQPoints = quadratureFormula.size();
        const size_t nFaceQPoints = faceQuadratureFormula.size();

        const size_t dofsPerCell = fe->dofs_per_cell;

        FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
        Vector<double> cellRhs(dofsPerCell);

        std::vector<types::global_dof_index> localDofIndices (dofsPerCell);

        FEValues<dim> feValues(*fe, quadratureFormula,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> feFaceValues(*fe, faceQuadratureFormula,
                               update_values | update_normal_vectors |
                               update_quadrature_points | update_JxW_values);

        const RightHandSide<dim> rightHandSide;
        std::vector<double> rhsValues(nQPoints);
        const Solution<dim> exactSolution;

        for (auto && cell : dofHandler.active_cell_iterators()) {
            cellMatrix = 0;
            cellRhs = 0;

            feValues.reinit(cell);

            rightHandSide.value_list(feValues.get_quadrature_points(),
                                     rhsValues);

            for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
                for (size_t i = 0; i < dofsPerCell; ++i) {
                    for (size_t j = 0; j < dofsPerCell; ++j) {
                        cellMatrix(i, j) += ((feValues.shape_grad(i, qPoint) *
                                              feValues.shape_grad(j, qPoint)
                                              +
                                              feValues.shape_value(i, qPoint) *
                                              feValues.shape_value(j, qPoint)) *
                                             feValues.JxW(qPoint)
                                             );
                    }
                    cellRhs(i) += (feValues.shape_value(i, qPoint) *
                                   rhsValues[qPoint] *
                                   feValues.JxW(qPoint));
                }
            }

            for (size_t faceNumber = 0; faceNumber < GeometryInfo<dim>::faces_per_cell;
                 ++faceNumber) {
                if (cell->face(faceNumber)->at_boundary() &&
                    (cell->face(faceNumber)->boundary_id() == 1)) {
                    feFaceValues.reinit(cell, faceNumber);

                    for (size_t qPoint = 0; qPoint < nFaceQPoints; ++qPoint) {
                        const double neumannValue =
                           ( exactSolution.gradient(feFaceValues.quadrature_point(qPoint)) *
                             feFaceValues.normal_vector(qPoint));
                        for (size_t i = 0; i < dofsPerCell; ++i) {
                            cellRhs(i) += (neumannValue *
                                           feFaceValues.shape_value(i, qPoint) *
                                           feFaceValues.JxW(qPoint));
                        }
                    }
                }
            }

            cell->get_dof_indices(localDofIndices);
            for (size_t i = 0; i < dofsPerCell; ++i) {
                for (size_t j = 0; j <dofsPerCell; ++j) {
                    systemMatrix.add(localDofIndices[i],
                                     localDofIndices[j],
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
                                                 Solution<dim>(),
                                                 boundaryValues);
        MatrixTools::apply_boundary_values(boundaryValues,
                                           systemMatrix,
                                           solution,
                                           systemRhs);
    }

    template <int dim>
    void HelmholtzProblem<dim>::solve()
    {
        SolverControl solverControl(1000, 1.0e-12);
        SolverCG<> solver(solverControl);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(systemMatrix, 1.2);

        solver.solve(systemMatrix, solution, systemRhs,
                     preconditioner);

        hangingNodeConstraints.distribute(solution);
    }

    template <int dim>
    void HelmholtzProblem<dim>::refineGrid()
    {
        switch (refinementMode) {
        case globalRefinement:
            triangulation.refine_global(1);
            break;

        case adaptiveRefinement:
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
            break;
        }

        default:
        {
            Assert(false, ExcNotImplemented());
            break;
        }

        }
    }

    template <int dim>
    void HelmholtzProblem<dim>::processSolution(const size_t cycle)
    {
        Vector<float> differencePerCell (triangulation.n_active_cells());
        VectorTools::integrate_difference (dofHandler,
                                           solution,
                                           Solution<dim>(),
                                           differencePerCell,
                                           QGauss<dim>(3),
                                           VectorTools::L2_norm);
        const double l2Error = VectorTools::compute_global_error (triangulation,
                                                                  differencePerCell,
                                                                  VectorTools::L2_norm);

        VectorTools::integrate_difference (dofHandler,
                                           solution,
                                           Solution<dim>(),
                                           differencePerCell,
                                           QGauss<dim>(3),
                                           VectorTools::H1_seminorm);
        const double h1Error = VectorTools::compute_global_error (triangulation,
                                                                  differencePerCell,
                                                                  VectorTools::H1_seminorm);

        const QTrapez<1> qTrapez;
        const QIterated<dim> qIterated(qTrapez, 5);
        VectorTools::integrate_difference (dofHandler,
                                           solution,
                                           Solution<dim>(),
                                           differencePerCell,
                                           qIterated,
                                           VectorTools::Linfty_norm);
        const double lInfinityError = VectorTools::compute_global_error(triangulation,
                                                                       differencePerCell,
                                                                       VectorTools::Linfty_norm);

        const size_t nActiveCells = triangulation.n_active_cells();
        const size_t nDofs = dofHandler.n_dofs();

        std::cout << "Cycle " << cycle << ':'
                  << std::endl
                  << "   Number of active cells:       "
                  << nActiveCells
                  << std::endl
                  << "   Number of degrees of freedom: "
                  << nDofs
                  << std::endl;

        convergenceTable.add_value("cycle", cycle);
        convergenceTable.add_value("cells", nActiveCells);
        convergenceTable.add_value("dofs", nDofs);
        convergenceTable.add_value("L2", l2Error);
        convergenceTable.add_value("H1", h1Error);
        convergenceTable.add_value("Linfinity", lInfinityError);
    }

    template <int dim>
    void HelmholtzProblem<dim>::run()
    {
        size_t nCycles;
        if (refinementMode == RefinementMode::globalRefinement) {
            nCycles = 5;
        }
        else {
            nCycles = 9;
        }

        for (size_t cycle = 0; cycle < nCycles; ++cycle) {
            if (cycle == 0) {
                GridGenerator::hyper_cube (triangulation, -1, 1);
                triangulation.refine_global(3);

                for (auto && cell : triangulation.cell_iterators()) {
                    for (size_t faceNumber = 0;
                         faceNumber < GeometryInfo<dim>::faces_per_cell;
                         ++faceNumber) {
                        if ((std::fabs(cell->face(faceNumber)->center()(0) - (-1)) < 1e-12)
                            ||
                            (std::fabs(cell->face(faceNumber)->center()(1) - (-1)) < 1e-12)) {
                            cell->face(faceNumber)->set_boundary_id(1);
                        }
                    }
                }
            }
            else {
                refineGrid();
            }

            setupSystem();

            assembleSystem();
            solve();

            processSolution(cycle);
        }

        std::string vtkFilename;
        switch (refinementMode) {
        case RefinementMode::globalRefinement:
            vtkFilename += "solution-global";
            break;
        case RefinementMode::adaptiveRefinement:
            vtkFilename += "solution-adaptive";
            break;
        default:
            Assert(false, ExcNotImplemented());
        }

        switch (fe->degree) {
        case 1:
            vtkFilename += "-q1";
            break;
        case 2:
            vtkFilename += "-q2";
            break;
        default:
            Assert(false, ExcNotImplemented());
        }

        vtkFilename += ".vtk";
        std::ofstream out(vtkFilename);

        DataOut<dim> dataOut;
        dataOut.attach_dof_handler(dofHandler);
        dataOut.add_data_vector(solution, "solution");

        dataOut.build_patches(fe->degree);
        dataOut.write_vtk(out);

        convergenceTable.set_precision("L2", 3);
        convergenceTable.set_precision("H1", 3);
        convergenceTable.set_precision("Linfinity", 3);

        convergenceTable.set_scientific("L2", true);
        convergenceTable.set_scientific("H1", true);
        convergenceTable.set_scientific("Linfinity", true);

        convergenceTable.set_tex_caption("cells", "\\# cells");
        convergenceTable.set_tex_caption("dofs", "\\# dofs");
        convergenceTable.set_tex_caption("L2", "L^2-error");
        convergenceTable.set_tex_caption("H1", "H^1-error");
        convergenceTable.set_tex_caption("Linfinity", "L^\\infty-error");

        convergenceTable.set_tex_format("cells", "r");
        convergenceTable.set_tex_format("dofs", "r");

        std::cout << std::endl;
        convergenceTable.write_text(std::cout);

        std::string errorFilename = "error";
        switch (refinementMode)
          {
          case RefinementMode::globalRefinement:
            errorFilename += "-global";
            break;
          case RefinementMode::adaptiveRefinement:
            errorFilename += "-adaptive";
            break;
          default:
            Assert (false, ExcNotImplemented());
          }
        switch (fe->degree)
          {
          case 1:
            errorFilename += "-q1";
            break;
          case 2:
            errorFilename += "-q2";
            break;
          default:
            Assert (false, ExcNotImplemented());
          }
        errorFilename += ".tex";
        std::ofstream error_table_file(errorFilename.c_str());
        convergenceTable.write_tex(error_table_file);


    }
}


int main()
{
    const size_t dim = 2;
    using namespace dealii;
    using namespace step7;
    try {
        {
          std::cout << "Solving with Q1 elements, adaptive refinement" << std::endl
                    << "=============================================" << std::endl
                    << std::endl;
          FE_Q<dim> fe(1);
          HelmholtzProblem<dim>
          helmholtz_problem_2d (fe, HelmholtzProblem<dim>::RefinementMode::adaptiveRefinement);
          helmholtz_problem_2d.run ();
          std::cout << std::endl;
        }
        {
          std::cout << "Solving with Q1 elements, global refinement" << std::endl
                    << "===========================================" << std::endl
                    << std::endl;
          FE_Q<dim> fe(1);
          HelmholtzProblem<dim>
          helmholtz_problem_2d (fe, HelmholtzProblem<dim>::RefinementMode::globalRefinement);
          helmholtz_problem_2d.run ();
          std::cout << std::endl;
        }
        {
          std::cout << "Solving with Q2 elements, global refinement" << std::endl
                    << "===========================================" << std::endl
                    << std::endl;
          FE_Q<dim> fe(2);
          HelmholtzProblem<dim>
          helmholtz_problem_2d (fe, HelmholtzProblem<dim>::RefinementMode::globalRefinement);
          helmholtz_problem_2d.run ();
          std::cout << std::endl;
        }
        {
          std::cout << "Solving with Q2 elements, adaptive refinement" << std::endl
                    << "===========================================" << std::endl
                    << std::endl;
          FE_Q<dim> fe(2);
          HelmholtzProblem<dim>
          helmholtz_problem_2d (fe, HelmholtzProblem<dim>::RefinementMode::adaptiveRefinement);
          helmholtz_problem_2d.run ();
          std::cout << std::endl;
        }
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
