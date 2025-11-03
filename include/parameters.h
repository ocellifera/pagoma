#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>

#include <limits>

namespace pagoma {

struct Parameters
{
  enum RealNumber
  {
    FLOAT,
    DOUBLE
  };

  enum Backend
  {
    GPU,
    CPU
  };

  unsigned int dim = 0;
  unsigned int degree = 0;
  RealNumber number = static_cast<RealNumber>(-1);
  Backend backend = static_cast<Backend>(-1);
  double timestep = 0.0;
};

class ParameterHandler
{
public:
  ParameterHandler()
  {
    prm.declare_entry("dim",
                      "2",
                      dealii::Patterns::Integer(1, 3),
                      "The problem dimension",
                      true);
    prm.declare_entry("degree",
                      "2",
                      dealii::Patterns::Integer(1, 6),
                      "The element degree",
                      true);
    prm.declare_entry("real number",
                      "float",
                      dealii::Patterns::Selection("float|double"),
                      "The real number type for the fields",
                      true);
    prm.declare_entry("backend",
                      "GPU",
                      dealii::Patterns::Selection("GPU|CPU"),
                      "The backend for which to run the code",
                      true);
    prm.declare_entry("time step",
                      "0.0",
                      dealii::Patterns::Double(0.0, DBL_MAX),
                      "The timestep",
                      true);
  };

  void populate(Parameters& parameters, const std::string& filename)
  {
    // Parse the input file, skipping undefine entries
    prm.parse_input(filename, "", true, true);

    // Read and assign the parameters
    parameters.dim = prm.get_integer("dim");
    parameters.degree = prm.get_integer("degree");
    parameters.number = prm.get("real number") == "float"
                          ? Parameters::RealNumber::FLOAT
                          : Parameters::RealNumber::DOUBLE;
    parameters.backend = prm.get("backend") == "GPU" ? Parameters::Backend::GPU
                                                     : Parameters::Backend::CPU;
    parameters.timestep = prm.get_double("time step");

    // Print the parameter descriptions
    std::cout << "\n\n";
    prm.print_parameters(std::cout,
                         dealii::ParameterHandler::OutputStyle::Description);
  };

private:
  dealii::ParameterHandler prm;
};

}
