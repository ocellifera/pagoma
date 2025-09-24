#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include "mesh_manager.h"

namespace pagoma {

template<unsigned int dim, unsigned int spacedim>
class DoFManager
{
public:
  DoFManager() = default;

  void reinit(const typename MeshManager<dim, spacedim>::Triangulation& tria,
              const dealii::FiniteElement<dim, spacedim>& fe)
  {
    dof_handler.reinit(tria);
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
  }

  const dealii::DoFHandler<dim, spacedim>& get_dof_handler() const
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
  dealii::DoFHandler<dim, spacedim> dof_handler;

  dealii::IndexSet locally_owned_dofs;

  dealii::IndexSet locally_relevant_dofs;
};
}
