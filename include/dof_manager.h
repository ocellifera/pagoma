#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include "mesh_manager.h"

namespace pagoma {

template<unsigned int dim>
class DoFManager
{
public:
  DoFManager() = default;

  void reinit(const typename MeshManager<dim>::Triangulation& tria,
              const dealii::FiniteElement<dim, dim>& fe)
  {
    dof_handler.reinit(tria);
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
  }

  const dealii::DoFHandler<dim, dim>& get_dof_handler() const
  {
    return dof_handler;
  }

  const dealii::IndexSet& get_locally_owned_dofs() const
  {
    return locally_owned_dofs;
  }

  const dealii::IndexSet& get_locally_relevant_dofs() const
  {
    return locally_relevant_dofs;
  }

private:
  dealii::DoFHandler<dim, dim> dof_handler;

  dealii::IndexSet locally_owned_dofs;

  dealii::IndexSet locally_relevant_dofs;
};
}
