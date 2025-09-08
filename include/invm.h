#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace GPU {
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
} // namespace GPU
