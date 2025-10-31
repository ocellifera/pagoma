#pragma once

#include <deal.II/base/memory_space.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

namespace pagoma {
namespace GPU {
/**
 * Evaluation of the inverted mass matrix at each quadrature point.
 */
template<unsigned int dim, unsigned int degree, typename number>
class InvmQuad
{
public:
  InvmQuad() = default;

  DEAL_II_HOST_DEVICE void operator()(
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, number>* fe_eval,
    const unsigned int q_point) const
  {
    fe_eval->submit_value(1.0, q_point);
  };

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

template<unsigned int dim, unsigned int degree, typename number>
class LocalInvm
{
public:
  LocalInvm() = default;

  DEAL_II_HOST_DEVICE void operator()(
    const typename dealii::Portable::MatrixFree<dim, number>::Data* data,
    const dealii::Portable::DeviceVector<number>& src,
    dealii::Portable::DeviceVector<number>& dst) const
  {
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(
      data);

    fe_eval.apply_for_each_quad_point(InvmQuad<dim, degree, number>());
    fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values);
    fe_eval.distribute_local_to_global(dst);
  };

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

template<unsigned int dim, unsigned int degree, typename number>
class Invm : public dealii::EnableObserverPointer
{
public:
  using DefaultVector =
    dealii::LinearAlgebra::distributed::Vector<number,
                                               dealii::MemorySpace::Default>;
  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<number,
                                               dealii::MemorySpace::Host>;

  Invm(dealii::Portable::MatrixFree<dim, number>* _data)
    : data(_data)
  {
    data->initialize_dof_vector(invm);
  };

  const DefaultVector& get_invm() const { return invm; }

  void compute()
  {
    DefaultVector dummy;

    data->initialize_dof_vector(dummy);

    invm = 0.0;
    LocalInvm<dim, degree, number> local_invm_operator;
    data->cell_loop(local_invm_operator, dummy, invm);

    number* invm_dev = invm.get_values();
    const number tolerance = 1.0e-12;

    Kokkos::parallel_for(
      invm.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        if (Kokkos::abs(invm_dev[i]) > tolerance) {
          invm_dev[i] = 1.0 / invm_dev[i];
        } else {
          invm_dev[i] = 1.0;
        }
      });
  };

private:
  dealii::Portable::MatrixFree<dim, number>* data;

  DefaultVector invm;
};
} // namespace GPU

namespace CPU {
template<unsigned int dim, unsigned int degree, typename number>
class Invm
{
public:
  using DefaultVector =
    dealii::LinearAlgebra::distributed::Vector<number,
                                               dealii::MemorySpace::Default>;
  using HostVector =
    dealii::LinearAlgebra::distributed::Vector<number,
                                               dealii::MemorySpace::Host>;

  Invm(dealii::MatrixFree<dim, number>* _data)
    : data(_data)
  {
    data->initialize_dof_vector(invm);
  };

  const HostVector& get_invm() const { return invm; }

  void compute()
  {
    dealii::FEEvaluation<dim, degree, degree + 1, 1, number> fe_eval(*data);

    for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell) {
      fe_eval.reinit(cell);
      for (const unsigned int q_point : fe_eval.quadrature_point_indices()) {
        fe_eval.submit_value(dealii::VectorizedArray<number>(1.0), q_point);
      }
      fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values);
      fe_eval.distribute_local_to_global(invm);
    }
    invm.compress(dealii::VectorOperation::add);

    const number tolerance = 1.0e-12;

    for (unsigned int i = 0; i < invm.locally_owned_size(); ++i) {
      if (invm.local_element(i) > tolerance) {
        invm.local_element(i) = 1.0 / invm.local_element(i);
      } else {
        invm.local_element(i) = 1.0;
      }
    }
  }

private:
  dealii::MatrixFree<dim, number>* data;

  HostVector invm;
};
} // namespace CPU
}
