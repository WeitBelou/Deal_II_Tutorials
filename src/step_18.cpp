/**
 * @file Quasistatic elasticity. Parallel computing.
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace step18
{
    using namespace dealii;

    template <int dim>
    struct PointHistory
    {
        SymmetricTensor<2, dim> oldStress;
    };

    template <int dim>
    SymmetricTensor<4, dim> getStressStrainTensor(const double lambda,
                                                  const double mu)
    {
        //Kroneker symbol
        auto d = [](int i, int j) {
            return (i == j) ? 1.0 : 0.0;
        };

        SymmetricTensor<4, dim> tmp;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < dim; ++k) {
                    for (int l = 0; l < dim; ++l) {
                        tmp[i][j][k][l] = mu * (d(i, k) * d(j, l) + d(i, l) * d(j, k))
                                          + lambda * d(i, j) * d(k, l);
                    }
                }
            }
        }
        return tmp;
    }

    template <int dim>
    inline SymmetricTensor<2, dim> getStrain(const FEValues<dim> & feValues,
                                             const size_t shapeFunction,
                                             const size_t qPoint)
    {
        SymmetricTensor<2, dim> tmp;

        for (int i = 0; i < dim; ++i) {
            tmp[i][i] = feValues.shape_grad_component(shapeFunction, qPoint, i)[i];
        }

        for (int i = 0; i < dim; ++i) {
            for (int j = i + 1; j < dim; ++j) {
                tmp[i][j] = (feValues.shape_grad_component(shapeFunction, qPoint, i)[j] +
                            feValues.shape_grad_component(shapeFunction, qPoint, j)[i]) / 2;
            }
        }

        return tmp;
    }

    template <int dim>
    inline SymmetricTensor<2, dim> getStrain(const std::vector<Tensor<1, dim>> & grad)
    {
        Assert(grad.size() == dim, ExcInternalError());

        SymmetricTensor<2, dim> tmp;

        for (int i = 0; i < dim; ++i) {
            tmp[i][i] = grad[i][i];
        }

        for (int i = 0; i < dim; ++i) {
            for (int j = i + 1; j < dim; ++j) {
                tmp[i][j] = (grad[i][j] + grad[j][i]) / 2;
            }
        }

        return tmp;
    }

    template <int dim>
    Tensor<2, dim> getRotationMatrix(const std::vector<Tensor<1, dim>> & gradU);

    template <>
    Tensor<2, 2> getRotationMatrix(const std::vector<Tensor<1, 2>> & gradU)
    {
        const double curl = gradU[1][0] - gradU[0][1];
        const double angle = std::atan(curl);
        const double t[2][2] = {{cos(angle), sin(angle)}, {-sin(angle), cos(angle)}};
        return Tensor<2, 2>(t);
    }

    template <>
    Tensor<2, 3> getRotationMatrix(const std::vector<Tensor<1, 3>> & gradU)
    {
        const Point<3> curl (gradU[2][1] - gradU[1][2],
                             gradU[0][2] - gradU[2][0],
                             gradU[1][0] - gradU[0][1]);
        const double angle = std::atan(curl.norm());

        if (angle < 1e-9)
        {
            static const double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            static const Tensor<2, 3> rot(rotation);
            return rot;
        }

        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double t = 1 - c;

        const Point<3> axis = curl / curl.norm();

        const double rotation[3][3] = {{
                                       t * axis[0] * axis[0] + c,
                                       t * axis[0] * axis[1] + s * axis[2],
                                       t * axis[0] * axis[2] - s * axis[1]
                                       },
                                       {
                                       t * axis[0] * axis[1] - s * axis[2],
                                       t * axis[1] * axis[1] + c,
                                       t * axis[1] * axis[2] + s * axis[0]
                                       },
                                       {
                                       t * axis[0] * axis[2] + s * axis[1],
                                       t * axis[1] * axis[1] - s * axis[0],
                                       t * axis[2] * axis[2] + c
                                       }
                                      };
        return Tensor<2, 3>(rotation);
    }

    template <int dim>
    class TopLevel
    {
    public:
        TopLevel ();
        ~TopLevel ();

        void run ();

    private:
        void createCoarseGrid ();
        void setupSystem ();

        void assembleSystem ();
        void solveTimestep ();
        size_t solveLinearProblem ();
        void outputResults () const;

        void doInitialTimestep ();
        void doTimestep ();

        void refineInitialGrid ();
        void moveMesh ();

        void setupQuadraturePointHistory ();
        void updateQuadraturePointHistory ();

        parallel::shared::Triangulation<dim> triangulation;
        FESystem<dim> fe;
        DoFHandler<dim> dofHandler;
        ConstraintMatrix hangingNodeConstraints;

        const QGauss<dim> quadratureFormula;
        std::vector<PointHistory<dim>> quadraturePointHistory;

        PETScWrappers::MPI::SparseMatrix systemMatrix;
        PETScWrappers::MPI::Vector systemRhs;

        Vector<double> incrementalDisplacement;

        double presentTime;
        double presentTimestep;
        double endTime;
        size_t timestepNo;

        MPI_Comm mpiCommunicator;
        const size_t nMpiProcesses;
        const size_t thisMpiProcess;
        ConditionalOStream pcout;

        std::vector<types::global_dof_index> localDofsPerProcessor;

        IndexSet locallyOwnedDofs;
        IndexSet locallyRelevantDofs;

        size_t nLocalCells;

        static const SymmetricTensor<4, dim> stressStrainTensor;
    };

    template <int dim>
    class BodyForce : public Function<dim>
    {
    public:
        BodyForce ();

        virtual void vector_value (const Point<dim> & p,
                                   Vector<double> & values) const override;
        virtual void vector_value_list (const std::vector<Point<dim>> & points,
                                        std::vector<Vector<double>> & valueList) const override;
    };

    template <int dim>
    BodyForce<dim>::BodyForce () : Function<dim>(dim)
    {
    }

    template <int dim>
    inline void BodyForce<dim>::vector_value (const Point <dim> & /* p */,
                                  Vector <double> &values) const
    {
        Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

        const double g = 9.81;
        const double rho = 7700;

        values = 0;
        values(dim - 1) = -rho * g;
    }

    template <int dim>
    void BodyForce<dim>::vector_value_list (const std::vector <Point <dim>> &points,
                                       std::vector <Vector <double>> &valueList) const
    {
        const size_t nPoints = points.size();

        Assert(valueList.size() == nPoints,
               ExcDimensionMismatch(valueList.size(), nPoints));

        for (size_t p = 0; p < nPoints; ++p) {
            BodyForce<dim>::vector_value(points[p], valueList[p]);
        }
    }

    template <int dim>
    class IncrementalBoundaryValues : public Function<dim>
    {
    public:
        IncrementalBoundaryValues (const double presentTime,
                                   const double presentTimestep);

        virtual void vector_value (const Point<dim> & p,
                                   Vector<double> & values) const;

        virtual void vector_value_list (const std::vector<Point<dim>> & points,
                                        std::vector<Vector<double>> & valueList) const;
    private:
        const double velocity;
        const double presentTime;
        const double presentTimestep;
    };

    template <int dim>
    IncrementalBoundaryValues<dim>::IncrementalBoundaryValues (
            const double presentTime, const double presentTimestep)
            : Function<dim>(dim),
              velocity(0.1),
              presentTime(presentTime),
              presentTimestep(presentTimestep)
    {}

    template <int dim>
    inline void IncrementalBoundaryValues<dim>::vector_value (const Point <dim> & /*p*/,
                                                       Vector <double> &values) const
    {
        Assert (values.size() == dim, ExcDimensionMismatch(values.size(), dim));

        values = 0;
        values(2) = -presentTimestep * velocity;
    }

    template <int dim>
    void IncrementalBoundaryValues<dim>::vector_value_list (
            const std::vector <Point <dim>> &points,
            std::vector <Vector <double>> &valueList) const
    {
        const size_t nPoints = points.size();

        Assert(valueList.size() == nPoints,
              ExcDimensionMismatch(valueList.size(), nPoints));

        for (size_t i = 0; i < nPoints; ++i) {
            IncrementalBoundaryValues<dim>::vector_value(points[i], valueList[i]);
        }
    }

    template <int dim>
    const SymmetricTensor<4, dim> TopLevel<dim>::stressStrainTensor =
            getStressStrainTensor<dim>(9.695e10, 7.617e10);

    template <int dim>
    TopLevel<dim>::TopLevel ()
    :
    triangulation(MPI_COMM_WORLD),
    fe (FE_Q<dim>(1), dim),
    dofHandler (triangulation),
    quadratureFormula(2),
    mpiCommunicator(MPI_COMM_WORLD),
    nMpiProcesses(Utilities::MPI::n_mpi_processes(mpiCommunicator)),
    thisMpiProcess(Utilities::MPI::this_mpi_process(mpiCommunicator)),
    pcout (std::cout, thisMpiProcess == 0)
    { }

    template <int dim>
    TopLevel<dim>::~TopLevel ()
    {
        dofHandler.clear();
    }

    template <int dim>
    void TopLevel<dim>::run ()
    {
        presentTime = 0;
        presentTimestep = 1;
        endTime = 10;
        timestepNo = 0;

        doInitialTimestep();

        while (presentTime < endTime) {
            doTimestep();
        }
    }

    template <int dim>
    void TopLevel<dim>::createCoarseGrid ()
    {
        const double innerRadius = 0.8;
        const double outerRadius = 1.0;

        GridGenerator::cylinder_shell(triangulation, 3, innerRadius, outerRadius);
        for (auto &&cell : triangulation.active_cell_iterators()) {
            for (size_t f = 0; f < GeometryInfo <dim>::faces_per_cell; ++f) {
                if (cell->face(f)->at_boundary()) {
                    const Point <dim> faceCenter = cell->face(f)->center();

                    if (faceCenter[2] == 0) {
                        cell->face(f)->set_boundary_id(0);
                    }
                    else if (faceCenter[2] == 3) {
                        cell->face(f)->set_boundary_id(1);
                    }
                    else if (faceCenter.norm() <
                             (innerRadius + outerRadius) / 2) {
                        cell->face(f)->set_boundary_id(2);
                    }
                    else {
                        cell->face(f)->set_boundary_id(3);
                    }
                }
            }
        }

        static const CylindricalManifold<dim> cylindricalManifold (2);

        triangulation.set_all_manifold_ids(0);
        triangulation.set_manifold (0, cylindricalManifold);
        triangulation.refine_global(1);

        setupQuadraturePointHistory();
    }

    template <int dim>
    void TopLevel<dim>::setupSystem ()
    {
        dofHandler.distribute_dofs(fe);
        locallyOwnedDofs = dofHandler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dofHandler, locallyRelevantDofs);
        nLocalCells =
        GridTools::count_cells_with_subdomain_association(triangulation,
                                                          triangulation.locally_owned_subdomain());
        localDofsPerProcessor = dofHandler.n_locally_owned_dofs_per_processor();

        hangingNodeConstraints.clear();
        DoFTools::make_hanging_node_constraints(dofHandler, hangingNodeConstraints);
        hangingNodeConstraints.close();

        DynamicSparsityPattern sparsityPattern (locallyRelevantDofs);
        DoFTools::make_sparsity_pattern(dofHandler, sparsityPattern,
                                        hangingNodeConstraints, false);
        SparsityTools::distribute_sparsity_pattern(sparsityPattern,
                                                   localDofsPerProcessor,
                                                   mpiCommunicator,
                                                   locallyRelevantDofs);
        systemMatrix.reinit(locallyOwnedDofs, locallyOwnedDofs,
                            sparsityPattern, mpiCommunicator);
        systemRhs.reinit(locallyOwnedDofs, mpiCommunicator);
        incrementalDisplacement.reinit(dofHandler.n_dofs());
    }

    template <int dim>
    void TopLevel<dim>::assembleSystem ()
    {
        systemRhs = 0;
        systemMatrix = 0;

        FEValues<dim> feValues (fe, quadratureFormula,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

        const size_t dofsPerCell = fe.dofs_per_cell;
        const size_t nQPoints = quadratureFormula.size();

        FullMatrix<double> cellMatrix (dofsPerCell, dofsPerCell);
        Vector<double> cellRhs (dofsPerCell);

        std::vector<types::global_dof_index> localDofIndices(dofsPerCell);

        BodyForce<dim> bodyForce;
        std::vector<Vector<double>> bodyForceValues (nQPoints, Vector<double>(dim));

        for (auto &&cell : dofHandler.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                cellMatrix = 0;
                cellRhs = 0;

                feValues.reinit(cell);

                for (size_t i = 0; i < dofsPerCell; ++i) {
                    for (size_t j = 0; j < dofsPerCell; ++j) {
                        for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
                            const SymmetricTensor<2, dim>
                            epsPhiI = getStrain(feValues, i, qPoint),
                            epsPhiJ = getStrain(feValues, j, qPoint);

                            cellMatrix(i, j) +=
                                    (epsPhiI * stressStrainTensor * epsPhiJ
                                     *
                                    feValues.JxW(qPoint));
                        }
                    }
                }

                const PointHistory<dim> * localQuadraturePointsData =
                        reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
                bodyForce.vector_value_list(feValues.get_quadrature_points(),
                                           bodyForceValues);

                for (size_t i = 0; i < dofsPerCell; ++i) {
                    const size_t componentI = fe.system_to_component_index(i).first;

                    for (size_t qPoint = 0; qPoint < nQPoints; ++qPoint) {
                        const SymmetricTensor<2, dim> & oldStress =
                                localQuadraturePointsData[qPoint].oldStress;

                        cellRhs(i) += (bodyForceValues[qPoint](componentI) *
                                       feValues.shape_value(i, qPoint)
                                       -
                                       oldStress *
                                       getStrain(feValues, i, qPoint)) *
                                       feValues.JxW(qPoint);
                    }
                }

                cell->get_dof_indices (localDofIndices);

                hangingNodeConstraints.
                        distribute_local_to_global(cellMatrix, cellRhs,
                                                   localDofIndices,
                                                   systemMatrix, systemRhs);
            }
        }

        systemMatrix.compress(VectorOperation::add);
        systemRhs.compress(VectorOperation::add);

        FEValuesExtractors::Scalar zComponent (dim - 1);
        std::map<types::global_dof_index, double> boundaryValues;

        VectorTools::interpolate_boundary_values(dofHandler, 0,
                                                ZeroFunction<dim>(dim),
                                                boundaryValues);

        VectorTools::interpolate_boundary_values(dofHandler, 1,
                                                IncrementalBoundaryValues<dim>(presentTime,
                                                                              presentTimestep),
                                                boundaryValues,
                                                fe.component_mask(zComponent));

        PETScWrappers::MPI::Vector tmp(locallyOwnedDofs, mpiCommunicator);
        MatrixTools::apply_boundary_values(boundaryValues, systemMatrix,
                                           tmp, systemRhs, false);
        incrementalDisplacement = tmp;
    }

    template <int dim>
    void TopLevel<dim>::solveTimestep ()
    {
        pcout << "    Assembling system ... " << std::flush;
        assembleSystem();
        pcout << " norm of rhs is " << systemRhs.l2_norm()
                                     << std::endl;

        const size_t nIterations = solveLinearProblem();

        pcout << "    Solver converged in " << nIterations
              << " iterations." << std::endl;

        pcout << "    Updating quadrature point data ... " << std::flush;
        updateQuadraturePointHistory();
        pcout << std::endl;
    }

    template <int dim>
    size_t TopLevel<dim>::solveLinearProblem ()
    {
        PETScWrappers::MPI::Vector
        distributedIncrementalDisplacement (locallyOwnedDofs, mpiCommunicator);

        distributedIncrementalDisplacement = incrementalDisplacement;

        SolverControl solverControl (dofHandler.n_dofs(),
                                     1.0e-16 * systemRhs.l2_norm());

        PETScWrappers::SolverCG cg (solverControl, mpiCommunicator);
        PETScWrappers::PreconditionBlockJacobi preconditioner(systemMatrix);

        cg.solve(systemMatrix, distributedIncrementalDisplacement,
                systemRhs, preconditioner);

        incrementalDisplacement = distributedIncrementalDisplacement;

        hangingNodeConstraints.distribute(incrementalDisplacement);

        return solverControl.last_step();
    }

    template <int dim>
    void TopLevel<dim>::outputResults () const
    {
        DataOut<dim> dataOut;
        dataOut.attach_dof_handler(dofHandler);

        std::vector<std::string> solutionNames;
        switch (dim)
        {
            case 1:
                solutionNames.push_back ("delta_x");
                break;
            case 2:
                solutionNames.push_back ("delta_x");
                solutionNames.push_back ("delta_y");
                break;
            case 3:
                solutionNames.push_back ("delta_x");
                solutionNames.push_back ("delta_y");
                solutionNames.push_back ("delta_z");
                break;
            default:
            Assert (false, ExcNotImplemented());
        }

        dataOut.add_data_vector(incrementalDisplacement, solutionNames);

        Vector<double> normOfStress (triangulation.n_active_cells());
        {
            for (auto &&cell : triangulation.active_cell_iterators()) {
                if (cell->is_locally_owned()) {
                    SymmetricTensor<2, dim> accumulatedStress;
                    for (size_t q = 0; q < quadratureFormula.size(); ++q) {
                        accumulatedStress +=
                                reinterpret_cast<PointHistory<dim>*>(cell->user_pointer())[q].oldStress;
                    }
                    normOfStress(cell->active_cell_index()) =
                            (accumulatedStress / quadratureFormula.size()).norm();
                }
                else {
                    normOfStress(cell->active_cell_index()) = -1e+20;
                }
            }
        }

        dataOut.add_data_vector(normOfStress, "norm_of_stress");

        std::vector<types::subdomain_id> partitionInt (triangulation.n_active_cells());
        GridTools::get_subdomain_association(triangulation, partitionInt);
        const Vector<double> partitioning (partitionInt.begin(), partitionInt.end());
        dataOut.add_data_vector(partitioning, "partitioning");

        dataOut.build_patches();

        std::string filename ("solution-");
        filename += Utilities::int_to_string(timestepNo, 4);
        filename += ".";
        filename += Utilities::int_to_string(thisMpiProcess, 3);
        filename += ".vtu";

        AssertThrow(nMpiProcesses < 1000, ExcNotImplemented());

        std::ofstream out (filename);
        dataOut.write_vtu(out);

        if (thisMpiProcess == 0) {
            std::vector<std::string> filenames;
            for (size_t i = 0; i < nMpiProcesses; ++i) {
                filenames.push_back("solution-" + Utilities::int_to_string(timestepNo, 4)
                                    + "." + Utilities::int_to_string(i, 3) + ".vtu");
            }
            const std::string visitMasterFilename ("solution-" + Utilities::int_to_string(timestepNo, 4) + ".visit");
            std::ofstream visitMaster (visitMasterFilename);
            dataOut.write_visit_record(visitMaster, filenames);

            const std::string
                    pvtuMasterFilename = ("solution-" +
                                            Utilities::int_to_string(timestepNo,4) +
                                            ".pvtu");
            std::ofstream pvtuMaster (pvtuMasterFilename.c_str());
            dataOut.write_pvtu_record (pvtuMaster, filenames);

            static std::vector<std::pair<double, std::string>> timesAndNames;
            timesAndNames.push_back(std::make_pair(presentTime, pvtuMasterFilename));
            std::ofstream pvdOut ("solution.pvd");
            dataOut.write_pvd_record(pvdOut, timesAndNames);
        }
    }

    template <int dim>
    void TopLevel<dim>::doInitialTimestep ()
    {
        presentTime += presentTimestep;
        ++timestepNo;

        pcout << "Timestep: " << timestepNo << " at time " << presentTime
              << std::endl;

        for (int cycle = 0; cycle < 2; ++cycle) {
            pcout << "Cycle " << cycle << ":" << std::endl;

            if (cycle == 0) {
                createCoarseGrid();
            }
            else {
                refineInitialGrid();
            }

            pcout << "    Number of active cells: "
                  << triangulation.n_active_cells()
                  << " (by partition: ";

            for (size_t p = 0; p < nMpiProcesses; ++p) {
                pcout << ((p == 0) ? (' ') : ('+'))
                      << (GridTools::count_cells_with_subdomain_association(triangulation, p));
            }
            pcout << ")" << std::endl;

            setupSystem();

            pcout << "    Number of degrees of fredom: " << dofHandler.n_dofs()
                  << " (by partition:";
            for (size_t p = 0; p < nMpiProcesses; ++p) {
                pcout << ((p == 0) ? (' ') : ('+'))
                      << (DoFTools::count_dofs_with_subdomain_association(dofHandler, p));
            }
            pcout << ")" << std::endl;

            solveTimestep();
        }

        moveMesh();
        outputResults();

        pcout << std::endl;
    }

    template <int dim>
    void TopLevel<dim>::doTimestep ()
    {
        presentTime += presentTimestep;
        ++timestepNo;
        pcout << "Timestep " << timestepNo << " at time " << presentTime
              << std::endl;
        if (presentTime > endTime) {
            presentTimestep -= (presentTime - endTime);
            presentTime = endTime;
        }

        solveTimestep();

        moveMesh();
        outputResults();

        pcout << std::endl;
    }

    template <int dim>
    void TopLevel<dim>::refineInitialGrid ()
    {
        Vector<float> errorPerCell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate(dofHandler,
                                          QGauss<dim - 1>(2),
                                          typename FunctionMap<dim>::type(),
                                          incrementalDisplacement,
                                          errorPerCell,
                                          ComponentMask(),
                                          0,
                                          MultithreadInfo::n_threads(),
                                          thisMpiProcess);
        const size_t nLocalCells = triangulation.n_locally_owned_active_cells();

        PETScWrappers::MPI::Vector
        distributedErrorPerCell (mpiCommunicator,
                                 triangulation.n_active_cells(),
                                 nLocalCells);
        for (size_t i = 0; i < errorPerCell.size(); ++i) {
            if (errorPerCell(i) != 0) {
                distributedErrorPerCell(i) = errorPerCell(i);
            }
        }
        distributedErrorPerCell.compress(VectorOperation::insert);

        errorPerCell = distributedErrorPerCell;
        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                        errorPerCell,
                                                        0.35, 0.03);
        triangulation.execute_coarsening_and_refinement();

        setupQuadraturePointHistory();
    }

    template <int dim>
    void TopLevel<dim>::moveMesh ()
    {
        pcout << "    Moving mesh ..." << std::endl;

        std::vector<bool> vertexTouched (triangulation.n_vertices(), false);
        for (auto &&cell: dofHandler.active_cell_iterators()) {
            for (size_t v = 0; v < GeometryInfo <dim>::vertices_per_cell; ++v) {
                if (vertexTouched[cell->vertex_index(v)] == false) {
                    vertexTouched[cell->vertex_index(v)] = true;

                    Point<dim> vertexDisplacement;
                    for (int d = 0; d < dim; ++d) {
                        vertexDisplacement[d] =
                                incrementalDisplacement(cell->vertex_dof_index(v, d));
                    }
                    cell->vertex(v) += vertexDisplacement;
                }
            }
        }
    }

    template <int dim>
    void TopLevel<dim>::setupQuadraturePointHistory ()
    {
        size_t ourCells = 0;
        for (auto &&cell : triangulation.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                ++ourCells;
            }
        }

        triangulation.clear_user_data();

        //Trick that shrink vector
        {
            std::vector<PointHistory<dim>> tmp;
            tmp.swap(quadraturePointHistory);
        }
        quadraturePointHistory.resize(ourCells * quadratureFormula.size());

        size_t historyIndex = 0;
        for (auto &&cell : triangulation.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                cell->set_user_pointer(&quadraturePointHistory[historyIndex]);
                historyIndex += quadratureFormula.size();
            }
        }

        Assert(historyIndex == quadraturePointHistory.size(),
               ExcInternalError());
    }

    template <int dim>
    void TopLevel<dim>::updateQuadraturePointHistory ()
    {
        FEValues<dim> feValues (fe, quadratureFormula,
                               update_values | update_gradients);

        std::vector<std::vector<Tensor<1, dim>>>
                displacementIncrementGrads (quadratureFormula.size(),
                                             std::vector<Tensor<1, dim>>(dim));

        for (auto &&cell : dofHandler.active_cell_iterators()) {
            if (cell->is_locally_owned()) {
                PointHistory<dim> * localQuadraturePointsHistory =
                        reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());

                Assert(localQuadraturePointsHistory >= &quadraturePointHistory.front(),
                      ExcInternalError());
                Assert(localQuadraturePointsHistory < &quadraturePointHistory.back(),
                      ExcInternalError());

                feValues.reinit(cell);
                feValues.get_function_gradients(incrementalDisplacement,
                                               displacementIncrementGrads);
                for (size_t q = 0; q < quadratureFormula.size(); ++q) {
                    const SymmetricTensor<2, dim> newStress
                            = (localQuadraturePointsHistory[q].oldStress
                              +
                              (stressStrainTensor *
                              getStrain(displacementIncrementGrads[q])));

                    const Tensor<2, dim> rotation =
                            getRotationMatrix(displacementIncrementGrads[q]);

                    const SymmetricTensor<2, dim> rotatedNewStress =
                        symmetrize(transpose(rotation) *
                               static_cast<Tensor<2, dim>>(newStress) *
                               rotation);
                    localQuadraturePointsHistory[q].oldStress = rotatedNewStress;
                }

            }
        }
    }
}

int main(int argc, char ** argv)
{
    try {
        using namespace dealii;
        using namespace step18;

        Utilities::MPI::MPI_InitFinalize mpiInitialization(argc, argv, 1);

        TopLevel<3> elasticProblem;
        elasticProblem.run();
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