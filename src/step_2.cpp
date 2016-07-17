/**
 * @file Define degrees of freedom
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <fstream>

using namespace dealii;

void makeGrid(Triangulation<2> & triangulation);
void distributeDofs(DoFHandler<2> & doFHandler);
void renumberDofs(DoFHandler<2> & doFHandler);

int main()
{
    Triangulation<2> triangulation;
    makeGrid(triangulation);

    DoFHandler<2> doFHandler(triangulation);
    distributeDofs(doFHandler);
    renumberDofs(doFHandler);

    return 0;
}

void makeGrid (Triangulation<2> &triangulation)
{
    const Point<2> center(1, 0);
    const double r = 0.5;
    const double R = 1.0;
    GridGenerator::hyper_shell(triangulation, center, r, R, 5);

    static const SphericalManifold<2> manifoldDescription(center);
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, manifoldDescription);

    for (size_t i = 0; i < 3; ++i) {
        for (auto cell : triangulation.active_cell_iterators()) {
            for (size_t v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) {
                const double d = center.distance(cell->vertex(v));

                if (std::fabs(d - r) < 1e-10) {
                    cell->set_refine_flag();
                    break;
                }
            }
        }
        triangulation.execute_coarsening_and_refinement();
    }
}

void distributeDofs (DoFHandler<2> &doFHandler)
{
    static const FE_Q<2> finiteElement(5);
    doFHandler.distribute_dofs(finiteElement);

    DynamicSparsityPattern dynamicSparsityPattern(doFHandler.n_dofs(), doFHandler.n_dofs());

    DoFTools::make_sparsity_pattern(doFHandler, dynamicSparsityPattern);

    SparsityPattern sparsityPattern;
    sparsityPattern.copy_from(dynamicSparsityPattern);

    std::ofstream out("SparsityPattern1.svg");
    sparsityPattern.print_svg(out);
}

void renumberDofs (DoFHandler<2> &doFHandler)
{
    DoFRenumbering::Cuthill_McKee(doFHandler);

    DynamicSparsityPattern dynamicSparsityPattern(doFHandler.n_dofs(), doFHandler.n_dofs());
    DoFTools::make_sparsity_pattern(doFHandler, dynamicSparsityPattern);

    SparsityPattern sparsityPattern;
    sparsityPattern.copy_from(dynamicSparsityPattern);

    std::ofstream out("SparsityPattern2.svg");
    sparsityPattern.print_svg(out);
}

