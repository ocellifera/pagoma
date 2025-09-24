#pragma once

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

// Parameters for the mobility and gradient energy
constexpr double mobility = 1.0;
constexpr double gradient_energy = 2.0;
constexpr unsigned total_increments = 10000;

/**
 * Evaluation of the Allen-Cahn operator at each quadrature point.
 */
template<unsigned int dim, unsigned int degree>
class AllenCahnOperatorQuad
{
public:
  DEAL_II_HOST_DEVICE AllenCahnOperatorQuad(double _timestep)
    : timestep(_timestep) {};

  DEAL_II_HOST_DEVICE void operator()(
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double>* fe_eval,
    const unsigned int q_point) const
  {
    auto value = fe_eval->get_value(q_point);
    auto gradient = fe_eval->get_gradient(q_point);

    auto double_well = 4.0 * value * (value - 1.0) * (value - 0.5);
    auto value_submission = value - timestep * mobility * double_well;
    auto gradient_submission =
      -timestep * mobility * gradient_energy * gradient;

    fe_eval->submit_value(value_submission, q_point);
    fe_eval->submit_gradient(gradient_submission, q_point);
  };

  double timestep = 0.0;

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

/**
 * Local evaluation of the Allen-Cahn operator
 */
template<unsigned int dim, unsigned int degree>
class LocalAllenCahnOperator
{
public:
  DEAL_II_HOST_DEVICE LocalAllenCahnOperator(double _timestep)
    : timestep(_timestep) {};

  DEAL_II_HOST_DEVICE void operator()(
    const typename dealii::Portable::MatrixFree<dim, double>::Data* data,
    const dealii::Portable::DeviceVector<double>& src,
    dealii::Portable::DeviceVector<double>& dst) const
  {
    dealii::Portable::FEEvaluation<dim, degree, degree + 1, 1, double> fe_eval(
      data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(dealii::EvaluationFlags::EvaluationFlags::values |
                     dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(
      AllenCahnOperatorQuad<dim, degree>(timestep));
    fe_eval.integrate(dealii::EvaluationFlags::EvaluationFlags::values |
                      dealii::EvaluationFlags::EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  };

  double timestep = 0.0;

  static constexpr unsigned int n_q_points =
    dealii::Utilities::pow(degree + 1, dim);

  static constexpr unsigned int n_local_dofs = n_q_points;
};

/**
 * Allen-Cahn operator
 */
template<unsigned int dim, unsigned int degree>
class AllenCahnOperator : public dealii::EnableObserverPointer
{
public:
  AllenCahnOperator(const dealii::DoFHandler<dim>& dof_handler,
                    const dealii::AffineConstraints<double>& constraints)
  {
    const dealii::MappingQ<dim> mapping(degree);
    typename dealii::Portable::MatrixFree<dim, double>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_JxW_values;
    const dealii::QGaussLobatto<1> quadrature(degree + 1);
    data.reinit(mapping, dof_handler, constraints, quadrature, additional_data);
  };

  void vmult(dealii::LinearAlgebra::distributed::
               Vector<double, dealii::MemorySpace::Default>& dst,
             const dealii::LinearAlgebra::distributed::
               Vector<double, dealii::MemorySpace::Default>& src,
             double timestep) const
  {
    dst = 0.0;
    LocalAllenCahnOperator<dim, degree> allen_cahn_operator(timestep);
    data.cell_loop(allen_cahn_operator, src, dst);
    data.copy_constrained_values(src, dst);
  };

  void initialize_dof_vector(
    dealii::LinearAlgebra::distributed::Vector<double,
                                               dealii::MemorySpace::Default>&
      vec) const
  {
    data.initialize_dof_vector(vec);
  };

  dealii::Portable::MatrixFree<dim, double>* get_matrix_free_data()
  {
    return &data;
  }

private:
  dealii::Portable::MatrixFree<dim, double> data;
};
