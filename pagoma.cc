#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/config.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iterator>
#include <random>
#include <type_traits>

#include "include/allen_cahn_operator.h"
#include "include/constraint_manager.h"
#include "include/dof_manager.h"
#include "include/invm.h"
#include "include/mesh_manager.h"
#include "include/timer.h"
#include "include/utilities.h"

template<unsigned int dim>
class InitialCondition : public dealii::Function<dim, double>
{
public:
  InitialCondition() = default;

  double value([[maybe_unused]] const dealii::Point<dim>& point,
               [[maybe_unused]] unsigned int component = 0) const override
  {
    double scalar_value = 0.5;

    // Add a random perturbation
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist_random(0.0, 1.0);
    scalar_value += dist_random(gen) * 0.1;

    return scalar_value;
  };
};

template<unsigned int dim, unsigned int degree>
class AllenCahnProblem
{
public:
  AllenCahnProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , timer(mpi_communicator)
    , mesh_manager(mpi_communicator)
    , fe(degree)
    , mapping(degree)
    , pcout(std::cout,
            dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {};

  void run()
  {
    timer.start_section("mesh generation");

    pagoma::Cube<dim> cube(1, 0.0, 50.0);
    mesh_manager.generate_triangulation(
      [&](typename pagoma::MeshManager<dim>::Triangulation& triangulation) {
        cube.generate(triangulation);
      });
    mesh_manager.refine(7);

    timer.end_section("mesh generation");

    timer.start_section("setup");
    setup_system();
    timer.end_section("setup");

    timer.start_section("initial condition");
    apply_initial_condition();
    timer.end_section("initial condition");

    pcout << "  Number of active cells: "
          << mesh_manager.get_triangulation()->n_global_active_cells() << "\n"
          << "  Number of degrees of freedom: "
          << dof_manager.get_dof_handler().n_dofs() << std::endl;

    pcout << "\nOutputting initial condition...\n" << std::flush;
    timer.start_section("output");
    output(0);
    timer.end_section("output");

    for (unsigned int increment = 1; increment <= total_increments;
         increment++) {
      timer.start_section("solve");
      solve();
      timer.end_section("solve");
      if (increment % 1000 == 0) {
        timer.start_section("output");
        output(increment);
        timer.end_section("output");
      }
    }
  };

private:
  void setup_system()
  {
    dof_manager.reinit(*mesh_manager.get_triangulation(), fe);
    constraint_manager.reinit(dof_manager);

    // Create the cpu and gpu matrix free data objects
    const dealii::MappingQ<dim> mapping(degree);
    const dealii::QGaussLobatto<1> quadrature(degree + 1);

    typename dealii::MatrixFree<dim, double>::AdditionalData
      cpu_additional_data;
    cpu_additional_data.mapping_update_flags = dealii::update_values |
                                               dealii::update_gradients |
                                               dealii::update_JxW_values;
    cpu_data.reinit(mapping,
                    dof_manager.get_dof_handler(),
                    constraint_manager.get_constraints(),
                    quadrature,
                    cpu_additional_data);

    typename dealii::Portable::MatrixFree<dim, double>::AdditionalData
      gpu_additional_data;
    gpu_additional_data.mapping_update_flags = dealii::update_values |
                                               dealii::update_gradients |
                                               dealii::update_JxW_values;
    gpu_data.reinit(mapping,
                    dof_manager.get_dof_handler(),
                    constraint_manager.get_constraints(),
                    quadrature,
                    gpu_additional_data);

    // Create the cpu and gpu invm objects and compute the invm
    cpu_invm = std::make_unique<CPU::Invm<dim, degree>>(&cpu_data);
    cpu_invm->compute();
    gpu_invm = std::make_unique<GPU::Invm<dim, degree>>(&gpu_data);
    gpu_invm->compute();

    system_matrix.reset(new AllenCahnOperator<dim, degree>(
      dof_manager.get_dof_handler(), constraint_manager.get_constraints()));

    ghost_solution_host.reinit(dof_manager.get_locally_owned_dofs(),
                               dof_manager.get_locally_relevant_dofs(),
                               mpi_communicator);
    system_matrix->initialize_dof_vector(new_solution);
    system_matrix->initialize_dof_vector(old_solution);
  };

  void apply_initial_condition()
  {
    dealii::VectorTools::interpolate(mapping,
                                     dof_manager.get_dof_handler(),
                                     InitialCondition<dim>(),
                                     ghost_solution_host);

    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
      dof_manager.get_locally_owned_dofs());
    rw_vector.import_elements(ghost_solution_host,
                              dealii::VectorOperation::insert);
    old_solution.import_elements(rw_vector, dealii::VectorOperation::insert);
    new_solution.import_elements(rw_vector, dealii::VectorOperation::insert);
  };

  void solve()
  {
    system_matrix->vmult(new_solution, old_solution);
    new_solution.scale(gpu_invm->get_invm());
    new_solution.swap(old_solution);
  };

  void output(unsigned int increment)
  {
    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
      dof_manager.get_locally_owned_dofs());
    rw_vector.import_elements(new_solution, dealii::VectorOperation::insert);
    ghost_solution_host.import_elements(rw_vector,
                                        dealii::VectorOperation::insert);

    constraint_manager.apply(ghost_solution_host);
    ghost_solution_host.update_ghost_values();

    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_manager.get_dof_handler());
    data_out.add_data_vector(ghost_solution_host, "solution");
    data_out.build_patches();

    dealii::DataOutBase::VtkFlags flags;
    flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", increment, mpi_communicator, 6);

    pcout << "  solution norm: " << ghost_solution_host.l2_norm() << std::endl;
  };

  MPI_Comm mpi_communicator;

  pagoma::Timer timer;

  pagoma::MeshManager<dim> mesh_manager;

  pagoma::DoFManager<dim, dim> dof_manager;

  pagoma::ConstraintManager<double> constraint_manager;

  const dealii::FE_Q<dim> fe;

  const dealii::MappingQ<dim> mapping;

  dealii::MatrixFree<dim, double> cpu_data;
  dealii::Portable::MatrixFree<dim, double> gpu_data;

  std::unique_ptr<GPU::Invm<dim, degree>> gpu_invm;
  std::unique_ptr<CPU::Invm<dim, degree>> cpu_invm;

  std::unique_ptr<AllenCahnOperator<dim, degree>> system_matrix;

  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
    ghost_solution_host;
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
    new_solution;
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
    old_solution;

  dealii::ConditionalOStream pcout;
};

int
main(int argc, char* argv[])
{
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(
      argc, argv, dealii::numbers::invalid_unsigned_int);
    AllenCahnProblem<2, 2> allen_cahn_problem;
    allen_cahn_problem.run();
  } catch (std::exception& exc) {
    std::cerr << "\n\n\nException on processing:\n"
              << exc.what() << "\nAborting!\n\n\n"
              << std::flush;
  } catch (...) {
    std::cerr << "\n\n\nUnknown exception!\n\nAborting!\n\n\n" << std::flush;
  }
}
