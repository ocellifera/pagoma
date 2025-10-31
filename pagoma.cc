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
#include <stdexcept>
#include <type_traits>

#include "include/allen_cahn_operator.h"
#include "include/constraint_manager.h"
#include "include/dof_manager.h"
#include "include/invm.h"
#include "include/mesh_manager.h"
#include "include/parameters.h"
#include "include/solution_manager.h"
#include "include/timer.h"
#include "include/utilities.h"

template<unsigned int dim, typename number>
class InitialCondition : public dealii::Function<dim, number>
{
public:
  InitialCondition() = default;

  number value([[maybe_unused]] const dealii::Point<dim>& point,
               [[maybe_unused]] unsigned int component = 0) const override
  {
    number scalar_value = 0.5;

    // Add a random perturbation
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<number> dist_random(0.0, 1.0);
    scalar_value += dist_random(gen) * 0.1;

    return scalar_value;
  };
};

template<unsigned int dim,
         unsigned int spacedim,
         unsigned int degree,
         typename number>
class AllenCahnProblem
{
public:
  AllenCahnProblem(pagoma::Parameters _parameters)
    : parameters(_parameters)
    , mpi_communicator(MPI_COMM_WORLD)
    , timer(mpi_communicator)
    , mesh_manager(mpi_communicator)
    , fe(degree)
    , mapping(degree)
    , host_solutions(1)
    , device_solutions(2)
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

    typename dealii::MatrixFree<dim, number>::AdditionalData
      cpu_additional_data;
    cpu_additional_data.mapping_update_flags = dealii::update_values |
                                               dealii::update_gradients |
                                               dealii::update_JxW_values;
    cpu_data.reinit(mapping,
                    dof_manager.get_dof_handler(),
                    constraint_manager.get_constraints(),
                    quadrature,
                    cpu_additional_data);

    typename dealii::Portable::MatrixFree<dim, number>::AdditionalData
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
    cpu_invm = std::make_unique<pagoma::CPU::Invm<dim, degree, number>>(&cpu_data);
    cpu_invm->compute();
    gpu_invm = std::make_unique<pagoma::GPU::Invm<dim, degree, number>>(&gpu_data);
    gpu_invm->compute();

    system_matrix.reset(new AllenCahnOperator<dim, degree, number>(
      dof_manager.get_dof_handler(), constraint_manager.get_constraints()));

    host_solutions.reinit(dof_manager, mpi_communicator);
    device_solutions.reinit(*system_matrix, 0);
    device_solutions.reinit(*system_matrix, 1);
  };

  void apply_initial_condition()
  {
    dealii::VectorTools::interpolate(mapping,
                                     dof_manager.get_dof_handler(),
                                     InitialCondition<dim, number>(),
                                     host_solutions.get_solution());

    dealii::LinearAlgebra::ReadWriteVector<number> rw_vector(
      dof_manager.get_locally_owned_dofs());
    rw_vector.import_elements(host_solutions.get_solution(),
                              dealii::VectorOperation::insert);
    device_solutions.get_solution(0).import_elements(rw_vector, dealii::VectorOperation::insert);
    device_solutions.get_solution(1).import_elements(rw_vector, dealii::VectorOperation::insert);
  };

  void solve()
  {
    system_matrix->vmult(device_solutions.get_solution(1), device_solutions.get_solution(0), parameters.timestep);
    device_solutions.get_solution(1).scale(gpu_invm->get_invm());
    device_solutions.get_solution(1).swap(device_solutions.get_solution(0));
  };

  void output(unsigned int increment)
  {
    dealii::LinearAlgebra::ReadWriteVector<number> rw_vector(
      dof_manager.get_locally_owned_dofs());
    rw_vector.import_elements(device_solutions.get_solution(1), dealii::VectorOperation::insert);
    host_solutions.get_solution().import_elements(rw_vector,
                                        dealii::VectorOperation::insert);

    constraint_manager.apply(host_solutions.get_solution());
    host_solutions.get_solution().update_ghost_values();

    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_manager.get_dof_handler());
    data_out.add_data_vector(host_solutions.get_solution(), "solution");
    data_out.build_patches();

    dealii::DataOutBase::VtkFlags flags;
    flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", increment, mpi_communicator, 6);

    pcout << "  solution norm: " << host_solutions.get_solution().l2_norm() << std::endl;
  };

  pagoma::Parameters parameters;

  MPI_Comm mpi_communicator;

  pagoma::Timer timer;

  pagoma::MeshManager<dim> mesh_manager;

  pagoma::DoFManager<dim, dim> dof_manager;

  pagoma::ConstraintManager<number> constraint_manager;

  const dealii::FE_Q<dim> fe;

  const dealii::MappingQ<dim> mapping;

  dealii::MatrixFree<dim, number> cpu_data;
  dealii::Portable::MatrixFree<dim, number> gpu_data;

  std::unique_ptr<pagoma::GPU::Invm<dim, degree, number>> gpu_invm;
  std::unique_ptr<pagoma::CPU::Invm<dim, degree, number>> cpu_invm;

  std::unique_ptr<AllenCahnOperator<dim, degree, number>> system_matrix;

  pagoma::SolutionManager<number, dealii::MemorySpace::Host> host_solutions;
  pagoma::SolutionManager<number, dealii::MemorySpace::Default> device_solutions;

  dealii::ConditionalOStream pcout;
};

template<unsigned int dim,
         unsigned int spacedim,
         unsigned int degree,
         typename number>
void
run_problem(const pagoma::Parameters& parameters)
{
  AllenCahnProblem<dim, spacedim, degree, number> problem(parameters);
  problem.run();
}

template<unsigned int dim, unsigned int spacedim, unsigned int degree>
void
dispatch_number(const pagoma::Parameters& parameters)
{
  switch (parameters.number) {
    case pagoma::Parameters::RealNumber::FLOAT:
      run_problem<dim, spacedim, degree, float>(parameters);
      break;
    case pagoma::Parameters::RealNumber::DOUBLE:
      run_problem<dim, spacedim, degree, double>(parameters);
      break;
    default:
      throw std::runtime_error("Unsupport real number type");
  }
}

template<unsigned int dim, unsigned int spacedim>
void
dispatch_degree(const pagoma::Parameters& parameters)
{
  switch (parameters.degree) {
    case 1:
      dispatch_number<dim, spacedim, 1>(parameters);
      break;
    case 2:
      dispatch_number<dim, spacedim, 2>(parameters);
      break;
    case 3:
      dispatch_number<dim, spacedim, 3>(parameters);
      break;
    case 4:
      dispatch_number<dim, spacedim, 4>(parameters);
      break;
    case 5:
      dispatch_number<dim, spacedim, 5>(parameters);
      break;
    case 6:
      dispatch_number<dim, spacedim, 6>(parameters);
      break;
    default:
      throw std::runtime_error("Unsupported degree");
  }
}

template<unsigned int dim>
void
dispatch_spacedim(const pagoma::Parameters& parameters)
{
  switch (parameters.spacedim) {
    case 1:
      dispatch_degree<dim, 1>(parameters);
      break;
    case 2:
      dispatch_degree<dim, 2>(parameters);
      break;
    case 3:
      dispatch_degree<dim, 3>(parameters);
      break;
    default:
      throw std::runtime_error("Unsupported spacedim");
  }
}

void
dispatch_dim(const pagoma::Parameters& parameters)
{
  switch (parameters.dim) {
    case 1:
      dispatch_spacedim<1>(parameters);
      break;
    case 2:
      dispatch_spacedim<2>(parameters);
      break;
    case 3:
      dispatch_spacedim<3>(parameters);
      break;
    default:
      throw std::runtime_error("Unsupported dim");
  }
}

int
main(int argc, char* argv[])
{
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(
      argc, argv, dealii::numbers::invalid_unsigned_int);

    // Grab the parameters
    pagoma::ParameterHandler parameter_handler;
    pagoma::Parameters parameters;
    parameter_handler.populate(parameters, "parameters.prm");

    dispatch_dim(parameters);

  } catch (std::exception& exc) {
    std::cerr << "\n\n\nException on processing:\n"
              << exc.what() << "\nAborting!\n\n\n"
              << std::flush;
  } catch (...) {
    std::cerr << "\n\n\nUnknown exception!\n\nAborting!\n\n\n" << std::flush;
  }
}
