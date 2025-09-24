#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_space.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "dof_manager.h"

namespace pagoma {

template<typename number>
class ConstraintManager
{
public:
  ConstraintManager() = default;

  template<unsigned int dim, unsigned int spacedim>
  void reinit(const DoFManager<dim, spacedim>& dof_manager)
  {
    constraints.clear();
    constraints.reinit(dof_manager.get_locally_owned_dofs(),
                       dof_manager.get_locally_relevant_dofs());
    dealii::DoFTools::make_hanging_node_constraints(
      dof_manager.get_dof_handler(), constraints);

    // TODO: Add more constraints here

    constraints.close();
  }

  const dealii::AffineConstraints<number>& get_constraints() const
  {
    return constraints;
  }

  template<typename memory_space>
  void apply(
    dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec)
  {
    constraints.distribute(vec);
  }

  template<typename memory_space>
  void set_zero(
    dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec)
  {
    constraints.set_zero(vec);
  }

private:
  dealii::AffineConstraints<number> constraints;
};
}
