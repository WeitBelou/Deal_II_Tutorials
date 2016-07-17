/**
 * @file simple grid examples
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <fstream>
using namespace dealii;
void first_grid ()
{
    Triangulation<2> triangulation;
    GridGenerator::hyper_cube (triangulation);
    triangulation.refine_global (4);
    std::ofstream out ("simpleRectGrid.eps");
    GridOut gridOut;
    gridOut.write_eps (triangulation, out);
    std::cout << "Grid written to simpleRectGrid.eps" << std::endl;
}
void second_grid ()
{
    const Point<2> center(1,0);
    const double R = 1.0;
    const double r = 0.5;
    const int numOfCells = 10;

    //Manifold define before triangulation in order to
    const SphericalManifold<2> manifoldDescription(center);

    Triangulation<2> triangulation;

    GridGenerator::hyper_shell (triangulation, center, r, R, numOfCells);

    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold (0, manifoldDescription);

    //Refine grid in five steps towards the inner circle of the domain
    for (size_t i = 0; i < 5; ++i)
    {
        //iterate through the cells
        for (auto cell : triangulation.active_cell_iterators())
        {
            //Check all vertices in every cell to mark it
            for (size_t v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
                const double d = center.distance(cell->vertex(v));
                if (std::fabs(d - r) < 1e-10) {
                    cell->set_refine_flag ();
                    break;
                }
            }
        }
        //Let all marked cells do refinement
        triangulation.execute_coarsening_and_refinement ();
    }
    std::ofstream file ("circularGrid.eps");
    GridOut gridOut;
    gridOut.write_eps(triangulation, file);
    std::cout << "Grid written to circularGrid.eps" << std::endl;
}

int main ()
{
    first_grid ();
    second_grid ();
}
