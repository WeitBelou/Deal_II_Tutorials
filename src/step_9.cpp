#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
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
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include <fstream>
#include <iostream>

namespace Step9
{
//begin namespace Step9

using namespace dealii;

namespace AdvectionProblem
{
//begin namespace GeneralSolver
template <int dim>
class GeneralSolver
{
public:
    GeneralSolver (const TensorFunction<1, dim> &advection_field,
                   const Function<dim> &boundary_values,
                   const Function<dim> &right_hand_side);
    ~GeneralSolver ();
    void run ();

private:
    void setup_system ();

    const SmartPointer<const TensorFunction<1, dim>> advection_field;
    const SmartPointer<const Function<dim>> boundary_values;
    const SmartPointer<const Function<dim>> right_hand_side;

    struct AssemblyScratchData
    {
        AssemblyScratchData (const FiniteElement<dim> &fe);
        AssemblyScratchData (const AssemblyScratchData &scratch_data);

        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
    };

    struct AssemblyCopyData
    {
        FullMatrix<double> cell_matrix;
        Vector<double> cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
    };

    void assemble_system ();

    void
    local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                           AssemblyScratchData &scratch_data,
                           AssemblyCopyData &copy_data);
    void copy_local_to_global (const AssemblyCopyData &copy_data);

    void solve ();
    void refine_grid ();
    void output_results (const size_t cycle) const;

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;

    FE_Q<dim> fe;

    ConstraintMatrix hanging_node_constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> system_rhs;
    Vector<double> solution;
};

/**
 * AssemblyScratchData constructor.
 * @param fe FiniteElement to use
 */
template <int dim>
GeneralSolver<dim>::AssemblyScratchData::AssemblyScratchData (const FiniteElement<dim> &fe)
    :
    fe_values (fe,
               QGauss<dim> (2),
               update_values | update_gradients |
               update_quadrature_points | update_JxW_values),
    fe_face_values (fe,
                    QGauss<dim - 1> (2),
                    update_values | update_quadrature_points |
                    update_JxW_values | update_normal_vectors)
{

}

/**
 * AssemblyScratchData copy constructor.
 * @param scratch_data
 */
template <int dim>
GeneralSolver<dim>::AssemblyScratchData::AssemblyScratchData (const GeneralSolver::AssemblyScratchData &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe (),
               scratch_data.fe_values.get_quadrature (),
               scratch_data.fe_values.get_update_flags ()),
    fe_face_values (scratch_data.fe_face_values.get_fe (),
                    scratch_data.fe_face_values.get_quadrature (),
                    scratch_data.fe_face_values.get_update_flags ())
{

}

/**
 * GeneralSolver constructor
 */
template <int dim>
GeneralSolver<dim>::GeneralSolver (const TensorFunction<1, dim> &advection_field,
                                   const Function<dim> &boundary_values,
                                   const Function<dim> &right_hand_side)
    :
    advection_field (&advection_field),
    boundary_values (&boundary_values),
    right_hand_side (&right_hand_side),
    dof_handler (triangulation),
    fe (1)
{

}

/**
 * GeneralSolver destructor.
 */
template <int dim>
GeneralSolver<dim>::~GeneralSolver ()
{
    dof_handler.clear ();
}

/**
 *
 */
template <int dim>
void GeneralSolver<dim>::run ()
{
    for (unsigned int cycle = 0; cycle < 6; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, -1, 1);
            triangulation.refine_global (4);
        }
        else
        {
            refine_grid ();
        }
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells ()
                  << std::endl;
        setup_system ();
        std::cout << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs ()
                  << std::endl;
        assemble_system ();
        solve ();
        output_results (cycle);
    }
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    std::ofstream output ("final-solution.vtk");
    data_out.write_vtk (output);
}

/**
 * Setup linear system.
 */
template <int dim>
void GeneralSolver<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    hanging_node_constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             hanging_node_constraints);
    hanging_node_constraints.close ();

    DynamicSparsityPattern dsp (dof_handler.n_dofs (), dof_handler.n_dofs ());
    DoFTools::make_sparsity_pattern (dof_handler,
                                     dsp,
                                     hanging_node_constraints,
                                     true);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs ());
    system_rhs.reinit (dof_handler.n_dofs ());
}

/**
 * Assemble linear system
 */
template <int dim>
void GeneralSolver<dim>::assemble_system ()
{
    WorkStream::run (dof_handler.begin_active (),
                     dof_handler.end (),
                     *this,
                     &GeneralSolver<dim>::local_assemble_system,
                     &GeneralSolver<dim>::copy_local_to_global,
                     AssemblyScratchData (fe),
                     AssemblyCopyData ());
    hanging_node_constraints.condense (system_matrix);
    hanging_node_constraints.condense (system_rhs);
}

template <int dim>
void
GeneralSolver<dim>::local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           GeneralSolver::AssemblyScratchData &scratch_data,
                                           GeneralSolver::AssemblyCopyData &copy_data)
{
    const size_t dofs_per_cell = fe.dofs_per_cell;

    const size_t n_q_points = scratch_data.fe_values.get_quadrature ().size ();

    const size_t
        n_face_q_points = scratch_data.fe_face_values.get_quadrature ().size ();

    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.cell_rhs.reinit (dofs_per_cell);
    copy_data.local_dof_indices.resize (dofs_per_cell);

    std::vector<double> rhs_values (n_q_points);
    std::vector<Tensor<1, dim>> advection_directions (n_q_points);
    std::vector<double> face_boundary_values (n_face_q_points);
    std::vector<Tensor<1, dim>> face_advection_directions (n_face_q_points);

    scratch_data.fe_values.reinit (cell);
    advection_field->value_list (scratch_data.fe_values.get_quadrature_points (),
                                 advection_directions);
    right_hand_side->value_list (scratch_data.fe_values.get_quadrature_points (),
                                 rhs_values);

    const double delta = 0.1 * cell->diameter ();

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                copy_data.cell_matrix (i, j) += (
                    (advection_directions[q_point] *
                     scratch_data.fe_values.shape_grad (j, q_point)
                     *
                     (scratch_data.fe_values.shape_value (i, q_point)
                      +
                      delta *
                      (advection_directions[q_point] *
                       scratch_data.fe_values.shape_grad (i, q_point))
                     )
                    )
                    *
                    scratch_data.fe_values.JxW (q_point));
            }
            copy_data.cell_rhs (i) += ((scratch_data.fe_values.shape_value (i, q_point) +
                                        delta *
                                        (advection_directions[q_point] *
                                         scratch_data.fe_values.shape_grad (i, q_point))) *
                                       rhs_values[q_point] *
                                       scratch_data.fe_values.JxW (q_point));
        }
    }

    for (size_t face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
        if (cell->face (face)->at_boundary ())
        {
            scratch_data.fe_face_values.reinit (cell, face);
            boundary_values->value_list (scratch_data.fe_face_values.get_quadrature_points (),
                                         face_boundary_values);
            advection_field->value_list (scratch_data.fe_face_values.get_quadrature_points (),
                                         face_advection_directions);

            for (size_t q = 0; q < n_face_q_points; ++q)
            {
                if (scratch_data.fe_face_values.normal_vector (q) * face_advection_directions[q]
                    < 0)
                {
                    for (size_t i = 0; i < dofs_per_cell; ++i)
                    {
                        for (size_t j = 0; j < dofs_per_cell; ++j)
                        {
                            copy_data.cell_matrix (i, j) -= (face_advection_directions[q] *
                                                             scratch_data.fe_face_values.normal_vector (q) *
                                                             scratch_data.fe_face_values.shape_value (i, q) *
                                                             scratch_data.fe_face_values.shape_value (j, q) *
                                                             scratch_data.fe_face_values.JxW (q));
                        }

                        copy_data.cell_rhs (i) -= (face_advection_directions[q] *
                                                   scratch_data.fe_face_values.normal_vector (q) *
                                                   face_boundary_values[q] *
                                                   scratch_data.fe_face_values.shape_value (i, q) *
                                                   scratch_data.fe_face_values.JxW (q));
                    }
                }
            }
        }
    }

    cell->get_dof_indices (copy_data.local_dof_indices);
}

template <int dim>
void
GeneralSolver<dim>::copy_local_to_global (const GeneralSolver::AssemblyCopyData &copy_data)
{
    for (size_t i = 0; i < copy_data.local_dof_indices.size (); ++i)
    {
        for (size_t j = 0; j < copy_data.local_dof_indices.size (); ++j)
        {
            system_matrix.add (copy_data.local_dof_indices[i],
                               copy_data.local_dof_indices[j],
                               copy_data.cell_matrix (i, j));
        }

        system_rhs (copy_data.local_dof_indices[i]) += copy_data.cell_rhs (i);
    }
}

template <int dim>
void GeneralSolver<dim>::solve ()
{
    SolverControl solver_control (1000, 1e-12);
    SolverBicgstab<> bicgstab (solver_control);
    PreconditionJacobi<> preconditioner;
    preconditioner.initialize (system_matrix, 1.0);
    bicgstab.solve (system_matrix, solution, system_rhs,
                    preconditioner);
    hanging_node_constraints.distribute (solution);
}

template <int dim>
void GeneralSolver<dim>::refine_grid ()
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells ());
    KellyErrorEstimator<2>::estimate (dof_handler,
                                      QGauss<dim - 1> (2),
                                      typename FunctionMap<dim>::type (),
                                      solution,
                                      estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.5, 0.03);
    triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void GeneralSolver<dim>::output_results (const size_t cycle) const
{
    std::string filename = "grid-";
    filename += ('0' + cycle);
    Assert (cycle < 10, ExcInternalError ());
    filename += ".eps";
    std::ofstream output (filename.c_str ());
    GridOut grid_out;
    grid_out.write_eps (triangulation, output);
}



//end namespace GeneralSolver
}

namespace DummyProblem
{
//begin namespace DummyProblem

template <int dim>
class AdvectionField: public TensorFunction<1, dim>
{
public:
    AdvectionField () : TensorFunction<1, dim, double> ()
    {}

    virtual Tensor<1, dim> value (const Point<dim> &p) const override;
    virtual void value_list (const std::vector<Point<dim>> &points,
                             std::vector<Tensor<1, dim>> &values) const override;
};

template <int dim>
Tensor<1, dim> AdvectionField<dim>::value (const Point<dim> &p) const
{
    Point <dim> value;
    value (1) = p (0);
    return value;
}

template <int dim>
void AdvectionField<dim>::value_list (const std::vector<Point<dim>> &points,
                                      std::vector<Tensor<1, dim>> &values) const
{
    Assert (values.size () == points.size (),
            ExcDimensionMismatch (values.size (), points.size ()))

    for (size_t i = 0; i < points.size (); ++i)
    {
        values[i] = AdvectionField<dim>::value (points[i]);
    }
}

template <int dim>
class RightHandSide: public Function<dim>
{
public:
    RightHandSide () : Function<dim> ()
    {}

    virtual double value (const Point<dim> &p,
                          const size_t component = 0) const override;
    virtual void value_list (const std::vector<Point<dim>> &points,
                             std::vector<double> &values,
                             const unsigned int component) const override;

};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &/*p*/,
                                  const size_t component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    return 0.0;
}

template <int dim>
void RightHandSide<dim>::value_list (const std::vector<Point<dim>> &points,
                                     std::vector<double> &values,
                                     const unsigned int component) const
{
    Assert (values.size () == points.size (),
            ExcDimensionMismatch (values.size (), points.size ()))
    for (size_t i = 0; i < values.size (); ++i)
    {
        values[i] = RightHandSide<dim>::value (points[i]);
    }
}

template <int dim>
class BoundaryValues: public Function<dim>
{
public:
    BoundaryValues () : Function<dim> ()
    {}
    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
    virtual void value_list (const std::vector<Point<dim>> &points,
                             std::vector<double> &values,
                             const unsigned int component = 0) const;
};
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                            const unsigned int component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));;
    return 0;
}

template <int dim>
void
BoundaryValues<dim>::value_list (const std::vector<Point<dim>> &points,
                                 std::vector<double> &values,
                                 const unsigned int component) const
{
    Assert (values.size () == points.size (),
            ExcDimensionMismatch (values.size (), points.size ()));
    for (unsigned int i = 0; i < points.size (); ++i)
    {
        values[i] = BoundaryValues<dim>::value (points[i], component);
    }
}

//end namespace DummyProblem
}

//end namespace Step9
}

int main ()
{
    try
    {
        dealii::MultithreadInfo::set_thread_limit ();
        Step9::DummyProblem::AdvectionField<2> advection_field;
        Step9::DummyProblem::BoundaryValues<2> boundary_values;
        Step9::DummyProblem::RightHandSide<2> right_hand_side;
        Step9::AdvectionProblem::GeneralSolver<2>
            advection_problem_2d (advection_field, boundary_values, right_hand_side);
        advection_problem_2d.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what () << std::endl
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
