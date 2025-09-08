#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/config.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// Parameters for the mobility and gradient energy
constexpr double mobility = 1.0;
constexpr double gradient_energy = 1.0;
constexpr double timestep = 1e-4;
constexpr unsigned total_increments = 10;

/**
 * Evaluation of the inverted mass matrix at each quadrature point.
 */
template <unsigned int dim, unsigned int degree> class InvmQuad {
public:
  InvmQuad() = default;

  DEAL_II_HOST_DEVICE void
  operator()(dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double>
                 *fe_eval,
             const unsigned int q_point) const {
    fe_eval->submit_value(1.0, q_point);
  };

  static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

/**
 * Evaluation of the Allen-Cahn operator at each quadrature point.
 */
template <unsigned int dim, unsigned int degree> class AllenCahnOperatorQuad {
public:
  AllenCahnOperatorQuad() = default;

  DEAL_II_HOST_DEVICE void
  operator()(dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double>
                 *fe_eval,
             const unsigned int q_point) const {
    auto value = fe_eval->get_value(q_point);
    auto gradient = fe_eval->get_gradient(q_point);

    auto double_well = 4.0 * value * (1.0 - value) * (value - 0.5);
    auto value_submission = value - timestep * mobility * double_well;
    auto gradient_submission =
        -timestep * mobility * gradient_energy * gradient;

    fe_eval->submit_value(value_submission, q_point);
    fe_eval->submit_gradient(gradient_submission, q_point);
  };

  static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

template <unsigned int dim, unsigned int degree> class LocalInvm {
public:
  LocalInvm() = default;

  DEAL_II_HOST_DEVICE void operator()(
      const typename dealii::Portable::MatrixFree<dim, double>::Data *data,
      const dealii::Portable::DeviceVector<double> &src,
      dealii::Portable::DeviceVector<double> &dst) const {
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double> fe_eval(
        data);

    fe_eval.apply_for_each_quad_point(InvmQuad<dim, degree>());
    fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(dst);
  };

  static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

/**
 * Local evaluation of the Allen-Cahn operator
 */
template <unsigned int dim, unsigned int degree> class LocalAllenCahnOperator {
public:
  LocalAllenCahnOperator() = default;

  DEAL_II_HOST_DEVICE void operator()(
      const typename dealii::Portable::MatrixFree<dim, double>::Data *data,
      const dealii::Portable::DeviceVector<double> &src,
      dealii::Portable::DeviceVector<double> &dst) const {
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double> fe_eval(
        data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(dealii::EvaluationFlags::EvaluationFlags::values |
                     dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(AllenCahnOperatorQuad<dim, degree>());
    fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values |
                      dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  };

  static constexpr unsigned int n_q_points =
      dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

/**
 * Allen-Cahn operator
 */
template <unsigned int dim, unsigned int degree>
class AllenCahnOperator : public dealii::EnableObserverPointer {
public:
  AllenCahnOperator(const dealii::DoFHandler<dim> &dof_handler,
                    const dealii::AffineConstraints<double> &constraints) {
    const dealii::MappingQ<dim> mapping(degree);
    typename dealii::Portable::MatrixFree<dim, double>::AdditionalData
        additional_data;
    additional_data.mapping_update_flags = dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_JxW_values;
    const dealii::QGaussLobatto<1> quadrature(degree + 1);
    data.reinit(mapping, dof_handler, constraints, quadrature, additional_data);
  };

  void vmult(dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::Default> &dst,
             const dealii::LinearAlgebra::distributed::Vector<
                 double, dealii::MemorySpace::Default> &src) const {
    dst = 0.0;
    LocalAllenCahnOperator<dim, degree> allen_cahn_operator;
    data.cell_loop(allen_cahn_operator, src, dst);
    data.copy_constrained_values(src, dst);
  };

  void initialize_dof_vector(dealii::LinearAlgebra::distributed::Vector<
                             double, dealii::MemorySpace::Default> &vec) const {
    data.initialize_dof_vector(vec);
  };

  dealii::Portable::MatrixFree<dim, double> *get_matrix_free_data() {
    return &data;
  }

private:
  dealii::Portable::MatrixFree<dim, double> data;
};

template <unsigned int dim, unsigned int degree>
class Invm : public dealii::EnableObserverPointer {
public:
  Invm(dealii::Portable::MatrixFree<dim, double> *_data) : data(_data) {};

  void compute(dealii::LinearAlgebra::distributed::Vector<
               double, dealii::MemorySpace::Default> &vec) {

    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Default>
        dummy;

    data->initialize_dof_vector(vec);
    data->initialize_dof_vector(dummy);

    vec = 0.0;
    LocalInvm<dim, degree> invm;
    data->cell_loop(invm, dummy, vec);

    double *vec_dev = vec.get_values();
    const double tolerance = 1.0e-12;

    Kokkos::parallel_for(
        vec.locally_owned_size(), KOKKOS_LAMBDA(int i) {
          if (Kokkos::abs(vec_dev[i]) > tolerance) {
            vec_dev[i] = 1.0 / vec_dev[i];
          } else {
            vec_dev[i] = 1.0;
          }
        });
  };

private:
  dealii::Portable::MatrixFree<dim, double> *data;
};

template <unsigned int dim>
class InitialCondition : public dealii::Function<dim, double> {
public:
  InitialCondition() = default;

  double value(const dealii::Point<dim> &point,
               unsigned int component = 0) const override {
    return 0.5 + 0.1 * std::sin(point[0]) * std::cos(point[1]);
  };
};

template <unsigned int dim, unsigned int degree> class AllenCahnProblem {
public:
  AllenCahnProblem()
      : mpi_communicator(MPI_COMM_WORLD), triangulation(mpi_communicator),
        fe(degree), dof_handler(triangulation), mapping(degree),
        pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(
                             mpi_communicator) == 0) {};

  void run() {
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 100.0);
    triangulation.refine_global(4);

    setup_system();
    compute_invm();
    apply_initial_condition();

    pcout << "  Number of active cells: "
          << triangulation.n_global_active_cells() << "\n"
          << "  Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    pcout << "\nOutputting initial condition...\n" << std::flush;
    output(0);

    for (unsigned int increment = 1; increment <= total_increments;
         increment++) {
      solve();

      if (increment % 1 == 0) {
        output(increment);
      }
    }
  };

private:
  void setup_system() {
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    system_matrix.reset(
        new AllenCahnOperator<dim, degree>(dof_handler, constraints));

    ghost_solution_host.reinit(locally_owned_dofs, locally_relevant_dofs,
                               mpi_communicator);
    system_matrix->initialize_dof_vector(new_solution);
    system_matrix->initialize_dof_vector(old_solution);
  };

  void compute_invm() {
    Invm<dim, degree> invm_computer(system_matrix->get_matrix_free_data());
    invm_computer.compute(invm);
  };

  void apply_initial_condition() {
    dealii::VectorTools::interpolate(
        mapping, dof_handler, InitialCondition<dim>(), ghost_solution_host);
  };

  void solve() {
    system_matrix->vmult(new_solution, old_solution);
    dealii::LinearAlgebra::ReadWriteVector<double> rw_vector(
        locally_owned_dofs);
    rw_vector.import_elements(new_solution, dealii::VectorOperation::insert);
    ghost_solution_host.import_elements(rw_vector,
                                        dealii::VectorOperation::insert);

    constraints.distribute(ghost_solution_host);

    ghost_solution_host.update_ghost_values();

    new_solution.swap(old_solution);
  };

  void output(unsigned int incremenet) const {
    dealii::DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ghost_solution_host, "solution");
    data_out.build_patches();

    dealii::DataOutBase::VtkFlags flags;
    flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./", "solution", incremenet,
                                        mpi_communicator, 6);

    pcout << "  solution norm: " << ghost_solution_host.l2_norm() << std::endl;
  };

  MPI_Comm mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  const dealii::FE_Q<dim> fe;

  dealii::DoFHandler<dim> dof_handler;

  const dealii::MappingQ<dim> mapping;

  dealii::IndexSet locally_owned_dofs;
  dealii::IndexSet locally_relevant_dofs;

  dealii::AffineConstraints<double> constraints;
  std::unique_ptr<AllenCahnOperator<dim, degree>> system_matrix;

  dealii::LinearAlgebra::distributed::Vector<double, dealii::MemorySpace::Host>
      ghost_solution_host;
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
      new_solution;
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
      old_solution;
  dealii::LinearAlgebra::distributed::Vector<double,
                                             dealii::MemorySpace::Default>
      invm;

  dealii::ConditionalOStream pcout;
};

int main(int argc, char *argv[]) {
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    AllenCahnProblem<2, 2> allen_cahn_problem;
    allen_cahn_problem.run();
  } catch (std::exception &exc) {
    std::cerr << "\n\n\nException on processing:\n"
              << exc.what() << "\nAborting!\n\n\n"
              << std::flush;
  } catch (...) {
    std::cerr << "\n\n\nUnknown exception!\n\nAborting!\n\n\n" << std::flush;
  }
}
