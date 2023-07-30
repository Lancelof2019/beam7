/* ---------------------------------------------------------------------
 * Copyright (C) 2010 - 2015 by the deal.II authors and
 *                              Jean-Paul Pelteret and Andrew McBride
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

/*
 * Authors: Jean-Paul Pelteret, University of Erlangen-Nuremberg,
 *          Andrew McBride, University of Cape Town, 2015, 2017
 */


// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.
#include <unistd.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_point_data.h>
#include <string>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <omp.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/config.h>
#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)
#include <deal.II/differentiation/ad.h>
#define ENABLE_SACADO_FORMULATION
#endif

// These must be included below the AD headers so that
// their math functions are available for use in the
// definition of tensors and kinematic quantities
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <Eigen/Sparse>

#include "gcmma/GCMMASolver.h"
#include "mma/MMASolver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
//#include <Eigen/Sparse>
//#include <Eigen/SparseLU>
// We then stick everything that relates to this tutorial program into a
// namespace of its own, and import all the deal.II function and class names
// into it:
//
//
//using namespace dealii;
//FullMatrix<double>  grad_cell_matrix;
namespace Cook_Membrane
{
  using namespace dealii;
//   extern FullMatrix<double>  grad_cell_matrix;
// @sect3{Run-time parameters}
//
// There are several parameters that can be set in the code so we set up a
// ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
// @sect4{Assembly method}

// Here we specify whether automatic differentiation is to be used to assemble
// the linear system, and if so then what order of differentiation is to be
// employed.
    struct AssemblyMethod
    {
      unsigned int automatic_differentiation_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void AssemblyMethod::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Assembly method");
      {
        prm.declare_entry("Automatic differentiation order", "0",
                          Patterns::Integer(0,2),
                          "The automatic differentiation order to be used in the assembly of the linear system.\n"
                          "# Order = 0: Both the residual and linearisation are computed manually.\n"
                          "# Order = 1: The residual is computed manually but the linearisation is performed using AD.\n"
                          "# Order = 2: Both the residual and linearisation are computed using AD.");
      }
      prm.leave_subsection();
    }

    void AssemblyMethod::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Assembly method");
      {
        automatic_differentiation_order = prm.get_integer("Automatic differentiation order");
      }
      prm.leave_subsection();
    }

// @sect4{Finite Element system}

// Here we specify the polynomial order used to approximate the solution.
// The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      
      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

// @sect4{Geometry}

// Make adjustments to the problem geometry and its discretisation.
    struct Geometry
    {
      unsigned int elements_per_edge;
      double       scale;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Elements per edge", "8",
                          Patterns::Integer(0),
                          "Number of elements per long edge of the beam");

        prm.declare_entry("Grid scale", "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        elements_per_edge = prm.get_integer("Elements per edge");
        scale = prm.get_double("Grid scale");
      }
      prm.leave_subsection();
    }

// @sect4{Materials}

// We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
// neo-Hookean material.
    struct Materials
    {
      double nu;
      double mu;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Poisson's ratio", "0.3",
                          Patterns::Double(-1.0,0.5),
                          "Poisson's ratio");

       // prm.declare_entry("Shear modulus", "0.4225e6",
       prm.declare_entry("Shear modulus", "1.0e6",
                          Patterns::Double(),
                          "Shear modulus");
      }
      prm.leave_subsection();
    }

    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        nu = prm.get_double("Poisson's ratio");
        mu = prm.get_double("Shear modulus");
      }
      prm.leave_subsection();
    }

// @sect4{Linear solver}

// Next, we choose both solver and preconditioner settings.  The use of an
// effective preconditioner is critical to ensure convergence when a large
// nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      std::string preconditioner_type;
      double      preconditioner_relaxation;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type", "CG",
                          Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual", "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry("Max iteration multiplier", "1",
                          Patterns::Double(0.0),
                          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type", "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation", "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
        preconditioner_type = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

// @sect4{Nonlinear solver}

// A Newton-Raphson scheme is used to solve the nonlinear system of governing
// equations.  We now define the tolerances and the maximum number of
// iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson", "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force", "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement", "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f = prm.get_double("Tolerance force");
        tol_u = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }

// @sect4{Time}

// Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1",
                          Patterns::Double(),
                          "End time");

        prm.declare_entry("Time step size", "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
    struct AllParameters :
      public AssemblyMethod,
      public FESystem,
      public Geometry,
      public Materials,
      public LinearSolver,
      public NonlinearSolver,
      public Time

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      AssemblyMethod::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      AssemblyMethod::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  }


// @sect3{Time class}

// A simple class to store time data. Its functioning is transparent so no
// discussion is necessary. For simplicity we assume a constant time step
// size.
  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t)
    {}

    virtual ~Time()
    {}

    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
  };

// @sect3{Compressible neo-Hookean material within a one-field formulation}

// As discussed in the literature and step-44, Neo-Hookean materials are a type
// of hyperelastic materials.  The entire domain is assumed to be composed of a
// compressible neo-Hookean material.  This class defines the behaviour of
// this material within a one-field formulation.  Compressible neo-Hookean
// materials can be described by a strain-energy function (SEF) $ \Psi =
// \Psi_{\text{iso}}(\overline{\mathbf{b}}) + \Psi_{\text{vol}}(J)
// $.
//
// The isochoric response is given by $
// \Psi_{\text{iso}}(\overline{\mathbf{b}}) = c_{1} [\overline{I}_{1} - 3] $
// where $ c_{1} = \frac{\mu}{2} $ and $\overline{I}_{1}$ is the first
// invariant of the left- or right-isochoric Cauchy-Green deformation tensors.
// That is $\overline{I}_1 :=\textrm{tr}(\overline{\mathbf{b}})$.  In this
// example the SEF that governs the volumetric response is defined as $
// \Psi_{\text{vol}}(J) = \kappa \frac{1}{4} [ J^2 - 1
// - 2\textrm{ln}\; J ]$,  where $\kappa:= \lambda + 2/3 \mu$ is
// the <a href="http://en.wikipedia.org/wiki/Bulk_modulus">bulk modulus</a>
// and $\lambda$ is <a
// href="http://en.wikipedia.org/wiki/Lam%C3%A9_parameters">Lame's first
// parameter</a>.
//
// The following class will be used to characterize the material we work with,
// and provides a central point that one would need to modify if one were to
// implement a different material model. For it to work, we will store one
// object of this type per quadrature point, and in each of these objects
// store the current state (characterized by the values or measures  of the
// displacement field) so that we can compute the elastic coefficients
// linearized around the current state.
  template <int dim,typename NumberType>
  class Material_Compressible_Neo_Hook_One_Field
  {
  public:
    Material_Compressible_Neo_Hook_One_Field(const double mu,
                                             const double nu)
      :
      kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
      c_1(mu / 2.0)
    {
      Assert(kappa > 0, ExcInternalError());
    }

    ~Material_Compressible_Neo_Hook_One_Field()
    {}

    // The first function is the total energy
    // $\Psi = \Psi_{\textrm{iso}} + \Psi_{\textrm{vol}}$.
    NumberType
    get_Psi(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    }

    // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
    // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
    SymmetricTensor<2,dim,NumberType>
    get_tau(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar)
    {
      // See Holzapfel p231 eq6.98 onwards
      return get_tau_vol(det_F) + get_tau_iso(b_bar);
    }

    // The fourth-order elasticity tensor in the spatial setting
    // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
    // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
    // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
    // \mathbf{C} \partial \mathbf{C}}$
    SymmetricTensor<4,dim,NumberType>
    get_Jc(const NumberType                        &det_F,
           const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return get_Jc_vol(det_F) + get_Jc_iso(b_bar);
    }

  private:
    // Define constitutive model parameters $\kappa$ (bulk modulus) and the
    // neo-Hookean model parameter $c_1$:
    const double kappa;
    const double c_1;

    // Value of the volumetric free energy
    NumberType
    get_Psi_vol(const NumberType &det_F) const
    {
      return (kappa / 4.0) * (det_F*det_F - 1.0 - 2.0*std::log(det_F));
    }

    // Value of the isochoric free energy
    NumberType
    get_Psi_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return c_1 * (trace(b_bar) - dim);
    }

    // Derivative of the volumetric free energy with respect to
    // $J$ return $\frac{\partial
    // \Psi_{\text{vol}}(J)}{\partial J}$
    NumberType
    get_dPsi_vol_dJ(const NumberType &det_F) const
    {
      return (kappa / 2.0) * (det_F - 1.0 / det_F);
    }

    // The following functions are used internally in determining the result
    // of some of the public functions above. The first one determines the
    // volumetric Kirchhoff stress $\boldsymbol{\tau}_{\textrm{vol}}$.
    // Note the difference in its definition when compared to step-44.
    SymmetricTensor<2,dim,NumberType>
    get_tau_vol(const NumberType &det_F) const
    {
      return NumberType(get_dPsi_vol_dJ(det_F) * det_F) * Physics::Elasticity::StandardTensors<dim>::I;
    }

    // Next, determine the isochoric Kirchhoff stress
    // $\boldsymbol{\tau}_{\textrm{iso}} =
    // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2,dim,NumberType>
    get_tau_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar(b_bar);
    }

    // Then, determine the fictitious Kirchhoff stress
    // $\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2,dim,NumberType>
    get_tau_bar(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return 2.0 * c_1 * b_bar;
    }

    // Second derivative of the volumetric free energy wrt $J$. We
    // need the following computation explicitly in the tangent so we make it
    // public.  We calculate $\frac{\partial^2
    // \Psi_{\textrm{vol}}(J)}{\partial J \partial
    // J}$
    NumberType
    get_d2Psi_vol_dJ2(const NumberType &det_F) const
    {
      return ( (kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
    }

    // Calculate the volumetric part of the tangent $J
    // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
    // definition when compared to step-44. The extra terms result from two
    // quantities in $\boldsymbol{\tau}_{\textrm{vol}}$ being dependent on
    // $\boldsymbol{F}$.
    SymmetricTensor<4,dim,NumberType>
    get_Jc_vol(const NumberType &det_F) const
    {
      // See Holzapfel p265
      return det_F
             * ( (get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F))*Physics::Elasticity::StandardTensors<dim>::IxI
                 - (2.0 * get_dPsi_vol_dJ(det_F))*Physics::Elasticity::StandardTensors<dim>::S );
    }

    // Calculate the isochoric part of the tangent $J
    // \mathfrak{c}_\textrm{iso}$:
    SymmetricTensor<4,dim,NumberType>
    get_Jc_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar(b_bar);
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso(b_bar);
      const SymmetricTensor<4, dim> tau_iso_x_I
        = outer_product(tau_iso,
                        Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso
        = outer_product(Physics::Elasticity::StandardTensors<dim>::I,
                        tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar();

      return (2.0 / dim) * trace(tau_bar)
             * Physics::Elasticity::StandardTensors<dim>::dev_P
             - (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso)
             + Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar
             * Physics::Elasticity::StandardTensors<dim>::dev_P;
    }

    // Calculate the fictitious elasticity tensor $\overline{\mathfrak{c}}$.
    // For the material model chosen this is simply zero:
    SymmetricTensor<4,dim,double>
    get_c_bar() const
    {
      return SymmetricTensor<4, dim>();
    }
  };

// @sect3{Quadrature point history}

// As seen in step-18, the <code> PointHistory </code> class offers a method
// for storing data at the quadrature points.  Here each quadrature point
// holds a pointer to a material description.  Thus, different material models
// can be used in different regions of the domain.  Among other data, we
// choose to store the Kirchhoff stress $\boldsymbol{\tau}$ and the tangent
// $J\mathfrak{c}$ for the quadrature points.
  template <int dim,typename NumberType>
  class PointHistory
  {
  public:
    PointHistory()
    {}

    virtual ~PointHistory()
    {}

    // The first function is used to create a material object and to
    // initialize all tensors correctly: The second one updates the stored
    // values and stresses based on the current deformation measure
    // $\textrm{Grad}\mathbf{u}_{\textrm{n}}$.
    void setup_lqp (const Parameters::AllParameters &parameters)
    {
      material.reset(new Material_Compressible_Neo_Hook_One_Field<dim,NumberType>(parameters.mu,
                     parameters.nu));
    }

    // We offer an interface to retrieve certain data.
    // This is the strain energy:
    NumberType
    get_Psi(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return material->get_Psi(det_F,b_bar);
    }

    // Here are the kinetic variables. These are used in the material and
    // global tangent matrix and residual assembly operations:
    // First is the Kirchhoff stress:
    SymmetricTensor<2,dim,NumberType>
    get_tau(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return material->get_tau(det_F,b_bar);
    }

    // And the tangent:
    SymmetricTensor<4,dim,NumberType>
    get_Jc(const NumberType                        &det_F,
           const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return material->get_Jc(det_F,b_bar);
    }

    // In terms of member functions, this class stores for the quadrature
    // point it represents a copy of a material type in case different
    // materials are used in different regions of the domain, as well as the
    // inverse of the deformation gradient...
  private:
    std::shared_ptr< Material_Compressible_Neo_Hook_One_Field<dim,NumberType> > material;
  };


// @sect3{Quasi-static compressible finite-strain solid}

  // Forward declarations for classes that will
  // perform assembly of the linear system.
  template <int dim,typename NumberType>
  struct Assembler_Base;
  template <int dim,typename NumberType>
  struct Assembler;

// The Solid class is the central class in that it represents the problem at
// hand. It follows the usual scheme in that all it really has is a
// constructor, destructor and a <code>run()</code> function that dispatches
// all the work to private functions of this class:
  template <int dim,typename NumberType>
  class Solid
  {
  public:
    Solid(const Parameters::AllParameters &parameters);

    virtual
    ~Solid();

    void
    run();
      
 // private:
  public:

    // We start the collection of member functions with one that builds the
    // grid:
    void
    make_grid();

    // Set up the finite element system to be solved:
    void
    system_setup();

    // Several functions to assemble the system and right hand side matrices
    // using multithreading. Each of them comes as a wrapper function, one
    // that is executed to do the work in the WorkStream model on one cell,
    // and one that copies the work done on this one cell into the global
    // object that represents it:
    void
    assemble_system(const BlockVector<double> &solution_delta);

    // We use a separate data structure to perform the assembly. It needs access
    // to some low-level data, so we simply befriend the class instead of
    // creating a complex interface to provide access as necessary.
    friend struct Assembler_Base<dim,NumberType>;
    friend struct Assembler<dim,NumberType>;

    // Apply Dirichlet boundary conditions on the displacement field
    void
    make_constraints(const int &it_nr);

    // Create and update the quadrature points. Here, no data needs to be
    // copied into a global object, so the copy_local_to_global function is
    // empty:
    void
    setup_qph();

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    void
    solve_nonlinear_timestep(BlockVector<double> &solution_delta);

    std::pair<unsigned int, double>
    solve_linear_system(BlockVector<double> &newton_update);

    // Solution retrieval as well as post-processing and writing data to file:
    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;

    void
    output_results() const;

    // Finally, some member variables that describe the current state: A
    // collection of the parameters used to describe the problem setup...
    const Parameters::AllParameters &parameters;

    // ...the volume of the reference and current configurations...
    double                           vol_reference;
    double                           vol_current;

    // ...and description of the geometry on which the problem is solved:
    Triangulation<dim>               triangulation;

    // Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                             time;
    TimerOutput                      timer;

    // A storage object for quadrature point information. As opposed to
    // step-18, deal.II's native quadrature point data manager is employed here.
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim,NumberType> > quadrature_point_history;

    // A description of the finite-element system including the displacement
    // polynomial degree, the degree-of-freedom handler, number of DoFs per
    // cell and the extractor objects used to retrieve information from the
    // solution vectors:
    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

    // Description of how the block-system is arranged. There is just 1 block,
    // that contains a vector DOF $\mathbf{u}$.
    // There are two reasons that we retain the block system in this problem.
    // The first is pure laziness to perform further modifications to the
    // code from which this work originated. The second is that a block system
    // would typically necessary when extending this code to multiphysics
    // problems.
    static const unsigned int        n_blocks = 1;
    static const unsigned int        n_components = dim;
    static const unsigned int        first_u_component = 0;

    enum
    {
      u_dof = 0
    };

    std::vector<types::global_dof_index>  dofs_per_block;

    // Rules for Gauss-quadrature on both the cell and faces. The number of
    // quadrature points on both cells and faces is recorded.
    const QGauss<dim>                qf_cell;
    const QGauss<dim - 1>            qf_face;
    const unsigned int               n_q_points;
    const unsigned int               n_q_points_f;

    // Objects that store the converged solution and right-hand side vectors,
    // as well as the tangent matrix. There is a AffineConstraints object used
    // to keep track of constraints.  We make use of a sparsity pattern
    // designed for a block system.
    AffineConstraints<double>        constraints;
    BlockSparsityPattern             sparsity_pattern;
    BlockSparseMatrix<double>        tangent_matrix;
    BlockVector<double>              system_rhs;
    BlockVector<double>              solution_n;

    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
        :
        norm(1.0), u(1.0)
      {}

      void reset()
      {
        norm = 1.0;
        u = 1.0;
      }
      void normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
      }

      double norm, u;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update,
           error_update_0, error_update_norm;

    // Methods to calculate error measures
    void
    get_error_residual(Errors &error_residual);

    void
    get_error_update(const BlockVector<double> &newton_update,
                     Errors &error_update);

    // Print information to screen in a pleasing way...
    static
    void
    print_conv_header();

    void
    print_conv_footer();

    void
    print_vertical_tip_displacement();
 //   friend void get_global_stiff(Solid<3,double> &solid,BlockSparseMatrix<double>&copy_global_stiff);

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

     //Vector<double>     density;
     //Vector<double>     gradient;
     //density.reinit(dof_handler_ref.n_dofs());
     //gradient.reinit(dof_handler_ref.n_dofs());
     //
     //
     //
    // void compute_density();
     //void compute_gradient();
    // std::vector<std::vector<NumberType>> density_gradient(triangulation.n_active_cells(), std::vector<NumberType>(dim, 0.0));
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//    Vector<double> d_gradients;
  //  void  compute_density_gradient();
  //
  //
  //
  public:
   Eigen::MatrixXd dense_matrix;

   Eigen::MatrixXd inv_dense_matrix;
   int nelx=0;
   int nely=0;
   int nelz=0;
  //  std::vector< unsigned int > repetitions;
   //typename Assembler_Base<dim,NumberType>::PerTaskData_ASM per_task_data;
 
  };

     
   //   void get_global_stiff(Solid<3,double> &solid,BlockSparseMatrix<double>&copy_global_stiff){
     
     // copy_global_stiff=solid.tangent_matrix;
      //return copy_global_stiff;
 // }
 
 



// @sect3{Implementation of the <code>Solid</code> class}

// @sect4{Public interface}

// We initialise the Solid class using data extracted from the parameter file.
  template <int dim,typename NumberType>
  Solid<dim,NumberType>::Solid(const Parameters::AllParameters &parameters)
    :
    parameters(parameters),
    vol_reference (0.0),
    vol_current (0.0),
    triangulation(Triangulation<dim>::maximum_smoothing),
    time(parameters.end_time, parameters.delta_t),
    timer(std::cout,
          TimerOutput::summary,
          TimerOutput::wall_times),
    degree(parameters.poly_degree),
    // The Finite Element System is composed of dim continuous displacement
    // DOFs.
    fe(FE_Q<dim>(parameters.poly_degree), dim), // displacement
    dof_handler_ref(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(first_u_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.quad_order),
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())
  {

  }

// The class destructor simply clears the data held by the DOFHandler
  template <int dim,typename NumberType>
  Solid<dim,NumberType>::~Solid()
  {
    dof_handler_ref.clear();
  }


// In solving the quasi-static problem, the time becomes a loading parameter,
// i.e. we increasing the loading linearly with time, making the two concepts
// interchangeable. We choose to increment time linearly using a constant time
// step size.
//
// We start the function with preprocessing, and then output the initial grid
// before starting the simulation proper with the first time (and loading)
// increment.
//
//
//
//
//
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
/*
template <int dim,typename NumberType>

void Solid<dim,NumberType>::compute_density()
{

  const FESystem<dim> testfe(FE_Q<dim>(1), dim);
  DoFHandler<dim> test_dof_handler_ref(triangulation);	
  //testfe(FE_Q<dim>(1), dim);
  //test_dof_handler_ref(triangulation);
  test_dof_handler_ref.distribute_dofs(testfe);
  Vector<double> test_density;
  test_density.reinit(triangulation.n_active_cells());
  typename DoFHandler<dim>::active_cell_iterator cell = test_dof_handler_ref.begin_active(),
                                                 endc = test_dof_handler_ref.end();
  int pec=0;
  for (; cell != endc; ++cell)
  {
    // Compute the grid density for each cell, e.g., the inverse of the cell volume
    test_density[cell->active_cell_index()] = 1.0 / cell->measure();
    std::cout<<"------------------------active_cell_index :"<<cell->active_cell_index()<<"----------------------------------"<<std::endl;
    std::cout<<"density of ["<<pec<<"]:"<<test_density[cell->active_cell_index()]<<std::endl;
    pec++;
  }
}
*/


//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
/*
template <int dim,typename NumberType>


void Solid<dim,NumberType>::compute_gradient()
{
  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
                                                 endc = dof_handler_ref.end();

  for (; cell != endc; cell++)
  {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; face++)
    {
      if (cell->face(face)->at_boundary())
      {
        // Compute the gradient at each boundary face by differencing the neighboring cell densities
        const typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face);
        gradient[cell->face(face)->index()] = density[neighbor->active_cell_index()] - density[cell->active_cell_index()];
      }
    }
  }
}
*/



//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
/*


template <int dim,typename NumberType>
void Solid<dim,NumberType>::compute_density_gradient()
    {
        std::vector<std::vector<NumberType>> density_gradient(triangulation.n_active_cells(), std::vector<NumberType>(dim, 0.0));

        typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active();
        const typename DoFHandler<dim>::active_cell_iterator endc = dof_handler_ref.end();

        for (; cell != endc; ++cell)
        {
            const unsigned int cell_index = cell->active_cell_index();

            const NumberType cell_density = 1.0 / cell->measure();

            for (unsigned int d = 0; d < dim; ++d)
            {
                const double h = cell-> diameter()/ 2.0;
                const NumberType density_plus_h = 1.0 / (cell->neighbor(d)->measure());
                const NumberType density_minus_h = 1.0 / (cell->neighbor(d)->neighbor(d)->measure());
                const NumberType density_gradient_d = (density_plus_h - density_minus_h) / (2.0 * h);

                density_gradient[cell_index][d] = density_gradient_d;
            }
        }

	std::cout<<"gradient"<<std::endl;
      for (const auto &row : density_gradient)
       {
           for (const auto &value : row)
          {
             std::cout << value << " ";
           }
            std::cout << std::endl;
        }
    }


*/

/*
template <int dim,typename NumberType>
void Solid<dim,NumberType>::compute_density_gradient(){
  DoFHandler<3> dof_handler(triangulation);
  FE_Q<dim> fe(1);
  //DoFHandler<3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  Vector<double> density_gradients(triangulation.n_active_cells());
  
  




  density_gradients(triangulation.n_active_cells());
  unsigned int cell_index = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    std::vector<Point<3>> cell_vertices(dofs_per_cell);
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      cell_vertices[i] = cell->vertex(i);

     // std::cout<<"---------------------------------------------------"<<std::endl;
       //   std::cout<<cell<<"["<<cell_index<<"]"<<"["<<cell_vertices[i][0]<<","<<cell_vertices[i][1]<<","<<cell_vertices[i][2]<<"]"<<std::endl;
          //std::cout<<cell<<"["<<j<<",0]"cell_vertices[j][0]<<std::endl;

    }
      //std::cout<<"---------------------------------------------------"<<std::endl;
    Point<3> gradient={0,0,0};

    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        if (j != k)
        { 
          //std::cout<<"---------------------------------------------------"<<std::endl;
	  //std::cout<<cell<<"["<<j<<"]"cell_vertices[j][0]<<std::endl;
	  //std::cout<<cell<<"["<<j<<",0]"cell_vertices[j][0]<<std::endl;
	  //std::cout<<cell<<"["<<j<<",1]"cell_vertices[j][1]<<std::endl;
	  //std::cout<<cell<<"["<<j<<",2]"cell_vertices[j][2]<<std::endl;
	 double  l1_coord_norm=std::abs(cell_vertices[j][0] - cell_vertices[k][0])+std::abs(cell_vertices[j][1] - cell_vertices[k][1])+std::abs(cell_vertices[j][2] - cell_vertices[k][2]);
         // gradient += (cell_vertices[j] - cell_vertices[k]) /(cell_vertices[j] - cell_vertices[k]).norm();
//	std::cout<<"["<<cell_index<<"]-l1_coord_norm"<<":["<<j<<"-"<<k<<"].l1_norm"<<l1_coord_norm<<std::endl;
  //      std::cout<<"["<<cell_index<<"]-value0:"<<abs(cell_vertices[j][0] - cell_vertices[k][0])<<std::endl;
//	std::cout<<"["<<cell_index<<"]-value1:"<<abs(cell_vertices[j][1] - cell_vertices[k][1])<<std::endl;
//	std::cout<<"["<<cell_index<<"]-value2:"<<abs(cell_vertices[j][2] - cell_vertices[k][2])<<std::endl;
	Point<3> temp_gradient;
	temp_gradient=cell_vertices[j] - cell_vertices[k];
	std::cout<<"-------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>temp_gradient[0]:"<<temp_gradient[0]<<std::endl;
	temp_gradient[0]=temp_gradient[0];//l1_coord_norm;
	std::cout<<"-------------------------->>>>>>>>>>>>>>>temp_gradient[0]/l1_norm"<<temp_gradient[0]<<std::endl;
	temp_gradient[1]=temp_gradient[1];//l1_coord_norm;
	temp_gradient[2]=temp_gradient[2];//l1_coord_norm;
	 // gradient += (cell_vertices[j] - cell_vertices[k])/l1_coord_norm;
	gradient=gradient+temp_gradient;
	//std::cout<<"The value of gradient in cell["<<cell_index<<"] gradient------>"<<"["<<gradient[0]<<","<<gradient[1]<<","<<gradient[2]<<"]"<<std::endl;
	
        }
      }
    }


    //std::cout<<"---------------------------------------------------"<<std::endl;
    std::cout<<"<<<<<<<<<<<<<<<<<<<<gradient["<<cell_index<<"]"<<"["<<gradient[0]<<","<<gradient[1]<<","<<gradient[2]<<"]>>>>>>>>>>>>>>>"<<std::endl;
    double l1_gradient_norm=std::abs(gradient[0])+std::abs(gradient[1])+std::abs(gradient[2]);
    //density_gradients(cell_index) = gradient.norm();
    density_gradients(cell_index) =l1_gradient_norm;
    ++cell_index;
    gradient={0,0,0};
  }

   d_gradients=density_gradients;

}

*/




//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::run()
  {
    make_grid();
    system_setup();
    output_results();
    time.increment();

    // We then declare the incremental solution update $\varDelta
    // \mathbf{\Xi}:= \{\varDelta \mathbf{u}\}$ and start the loop over the
    // time domain.
    //
    // At the beginning, we reset the solution update for this time step...
    BlockVector<double> solution_delta(dofs_per_block);
    while (time.current() <= time.end())
      {
        solution_delta = 0.0;

        // ...solve the current time step and update total solution vector
        // $\mathbf{\Xi}_{\textrm{n}} = \mathbf{\Xi}_{\textrm{n-1}} +
        // \varDelta \mathbf{\Xi}$...
        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;

        // ...and plot the results before moving on happily to the next time
        // step:
        output_results();
        time.increment();
      }

    // Lastly, we print the vertical tip displacement of the Cook cantilever
   
    //
    //>>>>>>>>>>>>>>>>>>>>change>>>>>>>>>>>>>>>>>>>>>>>>>>>
   // density.reinit(dof_handler_ref.n_dofs());
   //
  //  std::cout<<"dof_handler_ref.n_dofs():"<<dof_handler_ref.n_dofs()<<std::endl;
//   density.reinit(triangulation.n_active_cells());
    
   // gradient.reinit(dof_handler_ref.n_dofs());
   //std::cout<<"triangulation.n_active_cells():"<<triangulation.n_active_cells()<<std::endl;    
   //gradient.reinit(triangulation.n_active_cells());
   
   //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print_vertical_tip_displacement();

   // std::cout<<"Density claculation"<<std::endl;
  //  compute_density();

    //sleep(15);
   // std::cout<<"Gradient calculation"<<std::endl;
   // compute_gradient();
   //compute_density_gradient();
  }


// @sect3{Private interface}

// @sect4{Solid::make_grid}

// On to the first of the private member functions. Here we create the
// triangulation of the domain, for which we choose a scaled an anisotripically
// discretised rectangle which is subsequently transformed into the correct
// of the Cook cantilever. Each relevant boundary face is then given a boundary
// ID number.
//
// We then determine the volume of the reference configuration and print it
// for comparison.

  template <int dim>
  Point<dim> grid_y_transform (const Point<dim> &pt_in)
  {
    //const double &x = pt_in[0];
    //const double &y = pt_in[1];
//
  //  const double y_upper = 44.0 + (16.0/48.0)*x; // Line defining upper edge of beam
   // const double y_lower =  0.0 + (44.0/48.0)*x; // Line defining lower edge of beam
   // const double theta = y/44.0; // Fraction of height along left side of beam
   // const double y_upper = 44 + (44.0/88)*x; // Line defining upper edge of beam
   // const double y_lower =  0.0 + (44.0/88)*x; // Line defining lower edge of beam
   // const double theta = y/44; // Fraction of height along left side of beam
  //  const double y_transform = (1-theta)*y_lower + theta*y_upper; // Final transformation

    Point<dim> pt_out = pt_in;
   // pt_out[1] = y_transform;

    return pt_out;
  }

  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::make_grid()
  {
    // Divide the beam, but only along the x- and y-coordinate directions
    std::vector< unsigned int > repetitions(dim, parameters.elements_per_edge);
    // Only allow one element through the thickness
    // (modelling a plane strain condition)
  /*  if (dim == 3){
     repetitions[dim-1] = 4;
    
    //repetitions[0] = 0;
    //repetitions[1] = 2;
    //repetitions[2] = 2;
    }
    */
    repetitions[0] = 16;
    repetitions[1] = 8;
    repetitions[2] = 8;
    nelx=repetitions[0];
    nely=repetitions[1];
    nelz=repetitions[2];
   // std::cout<<"x:"<<nelx<<"y:"<<nely<<"z:"<<nelz<<std::endl;
//    const Point<dim> bottom_left = (dim == 3 ? Point<dim>(0.0, 0.0, -0.5) : Point<dim>(0.0, 0.0));
  //  const Point<dim> top_right = (dim == 3 ? Point<dim>(48.0, 44.0, 0.5) : Point<dim>(48.0, 44.0));
        const Point<dim> top_right =Point<dim>(0.0, 0.0, -10); 
        const Point<dim> bottom_left =Point<dim>(40,20, 10);
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              bottom_left,
                                              top_right);

    // Since we wish to apply a Neumann BC to the right-hand surface, we
    // must find the cell faces in this part of the domain and mark them with
    // a distinct boundary ID number.  The faces we are looking for are on the
    // +x surface and will get boundary ID 11.
    // Dirichlet boundaries exist on the left-hand face of the beam (this fixed
    // boundary will get ID 1) and on the +Z and -Z faces (which correspond to
    // ID 2 and we will use to impose the plane strain condition)
    const double tol_boundary = 1e-6;
    typename Triangulation<dim>::active_cell_iterator cell =
      triangulation.begin_active(), endc = triangulation.end();
/*
    for (; cell != endc; ++cell)
      for (unsigned int face = 0;
           face < GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face(face)->at_boundary() == true)
          {
            if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
              cell->face(face)->set_boundary_id(1); // -X faces
            else if (std::abs(cell->face(face)->center()[0] - 48.0) < tol_boundary)
              cell->face(face)->set_boundary_id(11); // +X faces
            else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 0.5) < tol_boundary)
              cell->face(face)->set_boundary_id(2); // +Z and -Z faces
          }
*/


for (; cell != endc; ++cell)
      for (unsigned int face = 0;
           face < GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face(face)->at_boundary() == true)
          {
            if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
              cell->face(face)->set_boundary_id(1); // -X faces
            else if (std::abs(cell->face(face)->center()[0] - 40.0) < tol_boundary)
              cell->face(face)->set_boundary_id(11); // +X faces
            else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 10) < tol_boundary)
              cell->face(face)->set_boundary_id(2); // +Z and -Z faces
          }


    // Transform the hyper-rectangle into the beam shape
    GridTools::transform(&grid_y_transform<dim>, triangulation);

    GridTools::scale(parameters.scale, triangulation);

    vol_reference = GridTools::volume(triangulation);
    vol_current = vol_reference;
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
  }


// @sect4{Solid::system_setup}

// Next we describe how the FE system is setup.  We first determine the number
// of components per block. Since the displacement is a vector component, the
// first dim components belong to it.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::system_setup()
  {
    timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(n_components, u_dof); // Displacement

    // The DOF handler is then initialised and we renumber the grid in an
    // efficient manner. We also record the number of DOFs per block.
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: " << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
              << std::endl;

    // Setup the sparsity pattern and tangent matrix
    tangent_matrix.clear();
    {
      const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];

      BlockDynamicSparsityPattern csp(n_blocks, n_blocks);

      csp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
      csp.collect_sizes();

      // Naturally, for a one-field vector-valued problem, all of the
      // components of the system are coupled.
      Table<2, DoFTools::Coupling> coupling(n_components, n_components);
      for (unsigned int ii = 0; ii < n_components; ++ii)
        for (unsigned int jj = 0; jj < n_components; ++jj)
          coupling[ii][jj] = DoFTools::always;
      DoFTools::make_sparsity_pattern(dof_handler_ref,
                                      coupling,
                                      csp,
                                      constraints,
                                      false);
      sparsity_pattern.copy_from(csp);
    }

    tangent_matrix.reinit(sparsity_pattern);

    // We then set up storage vectors
    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    solution_n.reinit(dofs_per_block);
    solution_n.collect_sizes();

    // ...and finally set up the quadrature
    // point history:
    setup_qph();

    timer.leave_subsection();
  }


// @sect4{Solid::setup_qph}
// The method used to store quadrature information is already described in
// step-18 and step-44. Here we implement a similar setup for a SMP machine.
//
// Firstly the actual QPH data objects are created. This must be done only
// once the grid is refined to its finest level.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    // Next we setup the initial quadrature point data. Note that when
    // the quadrature point data is retrieved, it is returned as a vector
    // of smart pointers.
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active(); cell != triangulation.end(); ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim,NumberType> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters);
      }
  }


// @sect4{Solid::solve_nonlinear_timestep}

// The next function is the driver method for the Newton-Raphson scheme. At
// its top we create a new vector to store the current Newton update step,
// reset the error storage objects and print solver header.
  template <int dim,typename NumberType>
  void
  Solid<dim,NumberType>::solve_nonlinear_timestep(BlockVector<double> &solution_delta)
  {
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
              << time.current() << "s" << std::endl;

    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    // We now perform a number of Newton iterations to iteratively solve the
    // nonlinear problem.  Since the problem is fully nonlinear and we are
    // using a full Newton method, the data stored in the tangent matrix and
    // right-hand side vector is not reusable and must be cleared at each
    // Newton step.  We then initially build the right-hand side vector to
    // check for convergence (and store this value in the first iteration).
    // The unconstrained DOFs of the rhs vector hold the out-of-balance
    // forces. The building is done before assembling the system matrix as the
    // latter is an expensive operation and we can potentially avoid an extra
    // assembly process by not assembling the tangent matrix when convergence
    // is attained.
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR;
         ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        // If we have decided that we want to continue with the iteration, we
        // assemble the tangent, make and impose the Dirichlet constraints,
        // and do the solve of the linearized system:
        make_constraints(newton_iteration);
        assemble_system(solution_delta);

        get_error_residual(error_residual);

        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        // We can now determine the normalised residual error and check for
        // solution convergence:
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u
            && error_residual_norm.u <= parameters.tol_f)
          {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();

            break;
          }

        const std::pair<unsigned int, double>
        lin_solver_output = solve_linear_system(newton_update);

        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;

        // We can now determine the normalised Newton update error, and
        // perform the actual update of the solution increment for the current
        // time step, update all quadrature point information pertaining to
        // this new displacement and stress state and continue iterating:
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        solution_delta += newton_update;

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "  "
                  << lin_solver_output.second << "  " << error_residual_norm.norm
                  << "  " << error_residual_norm.u << "  "
                  << "  " << error_update_norm.norm << "  " << error_update_norm.u
                  << "  " << std::endl;
      }

    // At the end, if it turns out that we have in fact done more iterations
    // than the parameter file allowed, we raise an exception that can be
    // caught in the main() function. The call <code>AssertThrow(condition,
    // exc_object)</code> is in essence equivalent to <code>if (!cond) throw
    // exc_object;</code> but the former form fills certain fields in the
    // exception object that identify the location (filename and line number)
    // where the exception was raised to make it simpler to identify where the
    // problem happened.
    AssertThrow (newton_iteration <= parameters.max_iterations_NR,
                 ExcMessage("No convergence in nonlinear solver!"));
  }


// @sect4{Solid::print_conv_header, Solid::print_conv_footer and Solid::print_vertical_tip_displacement}

// This program prints out data in a nice table that is updated
// on a per-iteration basis. The next two functions set up the table
// header and footer:
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::print_conv_header()
  {
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "    SOLVER STEP    "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     NU_NORM     "
              << " NU_U " << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;
  }



  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::print_conv_footer()
  {
    static const unsigned int l_width = 20;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
              << "v / V_0:\t" << vol_current << " / " << vol_reference
              << std::endl;
  }

// At the end we also output the result that can be compared to that found in
// the literature, namely the displacement at the upper right corner of the
// beam.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::print_vertical_tip_displacement()
  {
    static const unsigned int l_width = 44;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    // The measurement point, as stated in the reference paper, is at the midway
    // point of the surface on which the traction is applied.
/*    const Point<dim> soln_pt = (dim == 3 ? 
                                Point<dim>(48.0*parameters.scale, 52.0*parameters.scale, 0.5*parameters.scale) : 
                                Point<dim>(48.0*parameters.scale, 52.0*parameters.scale));
  */
const Point<dim> soln_pt = Point<dim>(40*parameters.scale, 20*parameters.scale, 10*parameters.scale);
    double vertical_tip_displacement = 0.0;
    double vertical_tip_displacement_check = 0.0;

    typename DoFHandler<dim>::active_cell_iterator cell =
      dof_handler_ref.begin_active(), endc = dof_handler_ref.end();
    for (; cell != endc; ++cell)
      {
        // if (cell->point_inside(soln_pt) == true)
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          if (cell->vertex(v).distance(soln_pt) < 1e-6)
            {
              // Extract y-component of solution at the given point
              // This point is coindicent with a vertex, so we can
              // extract it directly as we're using FE_Q finite elements
              // that have support at the vertices
              vertical_tip_displacement = solution_n(cell->vertex_dof_index(v,u_dof+1));

              // Sanity check using alternate method to extract the solution
              // at the given point. To do this, we must create an FEValues instance
              // to help us extract the solution value at the desired point
              const MappingQ<dim> mapping (parameters.poly_degree);
              const Point<dim> qp_unit = mapping.transform_real_to_unit_cell(cell,soln_pt);
              const Quadrature<dim> soln_qrule (qp_unit);
              AssertThrow(soln_qrule.size() == 1, ExcInternalError());
              FEValues<dim> fe_values_soln (fe, soln_qrule, update_values);
              fe_values_soln.reinit(cell);

              // Extract y-component of solution at given point
              std::vector< Tensor<1,dim> > soln_values (soln_qrule.size());
              fe_values_soln[u_fe].get_function_values(solution_n,
                                                       soln_values);
              vertical_tip_displacement_check = soln_values[0][u_dof+1];

              break;
            }
      }
    AssertThrow(vertical_tip_displacement > 0.0, ExcMessage("Found no cell with point inside!"))

    std::cout << "Vertical tip displacement: " << vertical_tip_displacement
              << "\t Check: " << vertical_tip_displacement_check
              << std::endl;
  }


// @sect4{Solid::get_error_residual}

// Determine the true residual error for the problem.  That is, determine the
// error in the residual for the unconstrained degrees of freedom.  Note that to
// do so, we need to ignore constrained DOFs by setting the residual in these
// vector components to zero.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);

    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_dof).l2_norm();
  }


// @sect4{Solid::get_error_udpate}

// Determine the true Newton update error for the problem
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::get_error_update(const BlockVector<double> &newton_update,
                                               Errors &error_update)
  {
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_dof).l2_norm();
  }



// @sect4{Solid::get_total_solution}

// This function provides the total solution, which is valid at any Newton step.
// This is required as, to reduce computational error, the total solution is
// only updated at the end of the timestep.
  template <int dim,typename NumberType>
  BlockVector<double>
  Solid<dim,NumberType>::get_total_solution(const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
  }


// @sect4{Solid::assemble_system}

  template <int dim,typename NumberType>
  struct Assembler_Base
  {
    virtual ~Assembler_Base() {}

    // Here we deal with the tangent matrix assembly structures. The
    // PerTaskData object stores local contributions.
    struct PerTaskData_ASM
    {
      const Solid<dim,NumberType>          *solid;
      FullMatrix<double>                   cell_matrix;
      FullMatrix<double>                   dev_cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;

      PerTaskData_ASM(const Solid<dim,NumberType> *solid)
        :
        solid (solid),
        cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
        cell_rhs(solid->dofs_per_cell),
	dev_cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
	//grad_cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
        local_dof_indices(solid->dofs_per_cell)
      {}

      void reset()
      {
        cell_matrix = 0.0;
        cell_rhs = 0.0;
	dev_cell_matrix=0.0;
      }
    };

    // On the other hand, the ScratchData object stores the larger objects such as
    // the shape-function values array (<code>Nx</code>) and a shape function
    // gradient and symmetric gradient vector which we will use during the
    // assembly.
    struct ScratchData_ASM
    {
      const BlockVector<double>               &solution_total;
      std::vector<Tensor<2, dim,NumberType> >  solution_grads_u_total;

      FEValues<dim>                fe_values_ref;
      FEFaceValues<dim>            fe_face_values_ref;

      std::vector<std::vector<Tensor<2, dim,NumberType> > >         grad_Nx;
      std::vector<std::vector<SymmetricTensor<2,dim,NumberType> > > symm_grad_Nx;

      ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                      const QGauss<dim> &qf_cell,
                      const UpdateFlags uf_cell,
                      const QGauss<dim-1> & qf_face,
                      const UpdateFlags uf_face,
                      const BlockVector<double> &solution_total)
        :
        solution_total(solution_total),
        solution_grads_u_total(qf_cell.size()),
        fe_values_ref(fe_cell, qf_cell, uf_cell),
        fe_face_values_ref(fe_cell, qf_face, uf_face),
        grad_Nx(qf_cell.size(),
                std::vector<Tensor<2,dim,NumberType> >(fe_cell.dofs_per_cell)),
        symm_grad_Nx(qf_cell.size(),
                     std::vector<SymmetricTensor<2,dim,NumberType> >
                     (fe_cell.dofs_per_cell))
      {}

      ScratchData_ASM(const ScratchData_ASM &rhs)
        :
        solution_total (rhs.solution_total),
        solution_grads_u_total(rhs.solution_grads_u_total),
        fe_values_ref(rhs.fe_values_ref.get_fe(),
                      rhs.fe_values_ref.get_quadrature(),
                      rhs.fe_values_ref.get_update_flags()),
        fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                           rhs.fe_face_values_ref.get_quadrature(),
                           rhs.fe_face_values_ref.get_update_flags()),
        grad_Nx(rhs.grad_Nx),
        symm_grad_Nx(rhs.symm_grad_Nx)
      {}

      void reset()
      {
        const unsigned int n_q_points = fe_values_ref.get_quadrature().size();
        const unsigned int n_dofs_per_cell = fe_values_ref.dofs_per_cell;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                    ExcInternalError());
            Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                    ExcInternalError());

            solution_grads_u_total[q_point] = Tensor<2,dim,NumberType>();
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                grad_Nx[q_point][k] = Tensor<2,dim,NumberType>();
                symm_grad_Nx[q_point][k] = SymmetricTensor<2,dim,NumberType>();
              }
          }
      }

    };

    // Of course, we still have to define how we assemble the tangent matrix
    // contribution for a single cell.
    void
    assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                             ScratchData_ASM &scratch,
                             PerTaskData_ASM &data)
    {
      // Due to the C++ specialization rules, we need one more
      // level of indirection in order to define the assembly
      // routine for all different number. The next function call
      // is specialized for each NumberType, but to prevent having
      // to specialize the whole class along with it we have inlined
      // the definition of the other functions that are common to
      // all implementations.
      assemble_system_tangent_residual_one_cell(cell, scratch, data);
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell; 
    //  std::cout<<"--------------------------------------assembler dofs_per_cell----------------------------------------"<<dofs_per_cell<<std::endl;
     // std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
   /*   for(int i=0;i<dofs_per_cell;i++){
	      for(int j=0;j<dofs_per_cell;j++){

                 std::cout<<data.dev_cell_matrix(i,j)<<" ";

	      }
	      std::cout<<std::endl;
      }
  */
     //  std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
      assemble_neumann_contribution_one_cell(cell, scratch, data);
    }

    // This function adds the local contribution to the system matrix.
    void
    copy_local_to_global_ASM(const PerTaskData_ASM &data)
    { 
     // std::cout<<"OK,will write into file"<<std::endl;
     // std::ofstream outputFile("output.txt");
      const AffineConstraints<double> &constraints = data.solid->constraints;
      BlockSparseMatrix<double> &tangent_matrix = const_cast<Solid<dim,NumberType> *>(data.solid)->tangent_matrix;
      BlockVector<double> &system_rhs =  const_cast<Solid<dim,NumberType> *>(data.solid)->system_rhs;

      constraints.distribute_local_to_global(
        data.cell_matrix, data.cell_rhs,
        data.local_dof_indices,
        tangent_matrix, system_rhs);



     //std::cout<<"rows:"<<tangent_matrix.n_block_rows()<<std::endl;
     //std::cout<<"cols:"<<tangent_matrix.n_block_cols()<<std::endl;

    //////////////////////////////////////////

/*
  if (outputFile.is_open()) {
  // Loop over each block
  for (unsigned int i = 0; i < tangent_matrix.n_block_rows(); ++i) {
    for (unsigned int j = 0; j < tangent_matrix.n_block_cols(); ++j) {
      // Access the block (i, j)
      const BlockSparseMatrix<double>::value_type &block = tangent_matrix.block(i, j);

      // Loop over the rows of the block
      for (unsigned int row = 0; row < block.n_rows(); ++row) {
        // Loop over the columns of the block
        for (unsigned int col = 0; col < block.n_cols(); ++col) {
          // Write the value at (row, col) to the file
          outputFile << block(row, col) << " ";
        }
        outputFile << std::endl;
      }
    }
  }

  // Close the file
  outputFile.close();
} else {
  std::cout << "Failed to open the output file!" << std::endl;
}

     //////////////////////////////////////////////

*/

    }

  protected:

    // This function needs to exist in the base class for
    // Workstream to work with a reference to the base class.
    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &/*cell*/,
                                              ScratchData_ASM &/*scratch*/,
                                              PerTaskData_ASM &/*data*/)
    {
      AssertThrow(false, ExcPureFunctionCalled());
    }

    void
    assemble_neumann_contribution_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           ScratchData_ASM &scratch,
                                           PerTaskData_ASM &data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int &n_q_points_f = data.solid->n_q_points_f;
      const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
      const Parameters::AllParameters &parameters = data.solid->parameters;
      const Time &time = data.solid->time;
      const FESystem<dim> &fe = data.solid->fe;
      const unsigned int &u_dof = data.solid->u_dof;

      // Next we assemble the Neumann contribution. We first check to see it the
      // cell face exists on a boundary on which a traction is applied and add
      // the contribution if this is the case.
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() == true
            && cell->face(face)->boundary_id() == 11)
          {
            scratch.fe_face_values_ref.reinit(cell, face);

            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                 ++f_q_point)
              {
                // We specify the traction in reference configuration.
                // For this problem, a defined total vertical force is applied
                // in the reference configuration.
                // The direction of the applied traction is assumed not to
                // evolve with the deformation of the domain.

                // Note that the contributions to the right hand side vector we
                // compute here only exist in the displacement components of the
                // vector.
                const double time_ramp = (time.current() / time.end());
               // const double magnitude  = (1.0/(16.0*parameters.scale*1.0*parameters.scale))*time_ramp; // (Total force) / (RHS surface area)
                
		const double magnitude  = (1.0/(16.0*parameters.scale*1.0*parameters.scale))*time_ramp;
		
		Tensor<1,dim> dir;
                dir[1] = 1.0;
                const Tensor<1, dim> traction  = magnitude*dir;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;

                    if (i_group == u_dof)
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const double Ni =
                          scratch.fe_face_values_ref.shape_value(i,
                                                                 f_q_point);
                        const double JxW = scratch.fe_face_values_ref.JxW(
                                             f_q_point);

                        data.cell_rhs(i) += (Ni * traction[component_i])
                                            * JxW;
                      }
                  }
              }
          }
    }

  };

  template <int dim>
  struct Assembler<dim,double> : Assembler_Base<dim,double>
  {
    typedef double NumberType;
    using typename Assembler_Base<dim,NumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim,NumberType>::PerTaskData_ASM;
   // FullMatrix<double> assembler_dev_cell_matrix(this->dofs_per_cell,this->dofs_per_cell);
    virtual ~Assembler() {}

    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM &scratch,
                                              PerTaskData_ASM &data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int &n_q_points = data.solid->n_q_points;
      const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
      const FESystem<dim> &fe = data.solid->fe;
      const unsigned int &u_dof = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;
      //assembler_dev_cell_matrix(dofs_per_cell,dofs_per_cell);
      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      const std::vector<std::shared_ptr<const PointHistory<dim,NumberType> > > lqph =
        const_cast<const Solid<dim,NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      // We first need to find the solution gradients at quadrature points
      // inside the current cell and then we update each local QP using the
      // displacement gradient:
      scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                         scratch.solution_grads_u_total);

      // Now we build the local cell stiffness matrix. Since the global and
      // local system matrices are symmetric, we can exploit this property by
      // building only the lower half of the local matrix and copying the values
      // to the upper half.
      //
      // In doing so, we first extract some configuration dependent variables
      // from our QPH history objects for the current quadrature point.
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const Tensor<2,dim,NumberType> &grad_u = scratch.solution_grads_u_total[q_point];
          const Tensor<2,dim,NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
          const NumberType               det_F = determinant(F);
          const Tensor<2,dim,NumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2,dim,NumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
          const Tensor<2,dim,NumberType> F_inv = invert(F);
          Assert(det_F > NumberType(0.0), ExcInternalError());

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const unsigned int k_group = fe.system_to_base_index(k).first.first;

              if (k_group == u_dof)
                {
                  scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                  scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
              else
                Assert(k_group <= u_dof, ExcInternalError());
            }

          const SymmetricTensor<2,dim,NumberType> tau = lqph[q_point]->get_tau(det_F,b_bar);
          const SymmetricTensor<4,dim,NumberType> Jc  = lqph[q_point]->get_Jc(det_F,b_bar);
          const Tensor<2,dim,NumberType> tau_ns (tau);
          double dev_mu=data.solid->parameters.mu;
          // Next we define some aliases to make the assembly process easier to
          // follow
          const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
          const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
          const double JxW = scratch.fe_values_ref.JxW(q_point);
          //assembler_dev_cell_matrix(dofs_per_cell,dofs_per_cell);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe.system_to_component_index(i).first;
              const unsigned int i_group     = fe.system_to_base_index(i).first.first;

              if (i_group == u_dof)
                data.cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;
              else
                Assert(i_group <= u_dof, ExcInternalError());

              for (unsigned int j = 0; j <= i; ++j)
                {
                  const unsigned int component_j = fe.system_to_component_index(j).first;
                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                  // This is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
                  // contribution. It comprises a material contribution, and a
                  // geometrical stress contribution which is only added along
                  // the local matrix diagonals:
                  if ((i_group == j_group) && (i_group == u_dof))
                    {
                      data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                                * symm_grad_Nx[j] * JxW;

		      data.dev_cell_matrix(i,j) += symm_grad_Nx[i]*(Jc-Physics::Elasticity::StandardTensors<dim>::dev_P * SymmetricTensor<4, dim>()* Physics::Elasticity::StandardTensors<dim>::dev_P)/dev_mu * symm_grad_Nx[j] * JxW;
	//	      assembler_dev_cell_matrix(i,j)=data.dev_cell_matrix(i,j);
                    //  grad_cell_matrix(i,j)=data.dev_cell_matrix(i,j);
		      
		      if (component_i == component_j) // geometrical stress contribution
                        data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                  * grad_Nx[j][component_j] * JxW;
		      data.dev_cell_matrix(i,j) += grad_Nx[i][component_i] * tau/dev_mu * grad_Nx[j][component_j] * JxW;
	//	      assembler_dev_cell_matrix(i,j)=data.dev_cell_matrix(i,j);
                      //grad_cell_matrix(i,j)=data.dev_cell_matrix(i,j);
                    }
                  else
                    Assert((i_group <= u_dof) && (j_group <= u_dof),
                           ExcInternalError());
                }
            }
        }

        //    std::cout<<"-------------------------dev_cell_matrix--------------------------------"<<std::endl;
       //  for (unsigned int i = 0; i < dofs_per_cell; ++i){
       //	for (unsigned int j = 0; j < dofs_per_cell; ++j){
           //      if(j!=dofs_per_cell){
          //	   std::cout<<data.cell_matrix(i, j)<<" ";
        //
       //        }
        //	}
        //    std::cout<<std::endl;
      //}
     // std::cout<<"-------------------------dev_cell_matrix--------------------------------"<<std::endl;


      // Finally, we need to copy the lower half of the local matrix into the
      // upper half:
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j){
          data.cell_matrix(i, j) = data.cell_matrix(j, i);
	  data.dev_cell_matrix(i,j)=data.dev_cell_matrix(j,i);
	 // assembler_dev_cell_matrix(i,j)=assembler_dev_cell_matrix(j,i);
	  //grad_cell_matrix(i,j)=grad_cell_matrix(j,i);
	}


          std::ofstream outputFile("cell_matrix_output.txt");

    // Loop through the dev_cell_matrix and write its values to the file
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            if (j != dofs_per_cell - 1) {
               // std::cout << data.dev_cell_matrix(i, j) << " "; // Print to console
                outputFile << data.cell_matrix(i, j) << " "; // Write to file
            } else {
               // std::cout << data.dev_cell_matrix(i, j) << "\n"; // Print to console
                outputFile << data.cell_matrix(i, j) << "\n"; // Write to file
            }
        }
    }
         outputFile.close();



         /* std::cout<<"-------------------------dev_cell_matrix--------------------------------"<<std::endl;
        for (unsigned int i = 0; i < dofs_per_cell; ++i){
             for (unsigned int j = 0; j < dofs_per_cell; ++j){
                if(j!=dofs_per_cell){
                 std::cout<<data.dev_cell_matrix(i, j)<<" ";
              }
             }
          std::cout<<std::endl;
      }
      std::cout<<"-------------------------dev_cell_matrix--------------------------------"<<std::endl;
*/


/*
  for (unsigned int i = 0; i < dofs_per_cell; ++i){
             for (unsigned int j = 0; j < dofs_per_cell; ++j){
                 if(j==dofs_per_cell-1){
                    std:: cout<<data.cell_matrix(i,j)<<"|"<<std::endl;
                }
                 else if(j==0)
                       cout<<"|"<<data.cell_matrix(i,j)<<" ";
					  else
					   cout<<data.cell_matrix(i,j)<<" ";
         
               
		  }
		}

*/
//      std::cout<<dofs_per_cell<<std::endl;

    }

  };

#ifdef ENABLE_SACADO_FORMULATION


  template <int dim>
  struct Assembler<dim,Sacado::Fad::DFad<double> > : Assembler_Base<dim,Sacado::Fad::DFad<double> >
  {
    typedef Sacado::Fad::DFad<double> ADNumberType;
    using typename Assembler_Base<dim,ADNumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim,ADNumberType>::PerTaskData_ASM;

    virtual ~Assembler() {}

    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM &scratch,
                                              PerTaskData_ASM &data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int &n_q_points = data.solid->n_q_points;
      const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
      const FESystem<dim> &fe = data.solid->fe;
      const unsigned int &u_dof = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType> > > lqph =
        const_cast<const Solid<dim,ADNumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      const unsigned int n_independent_variables = data.local_dof_indices.size();
      std::vector<double> local_dof_values(n_independent_variables);
      cell->get_dof_values(scratch.solution_total,
                           local_dof_values.begin(),
                           local_dof_values.end());

      // We now retrieve a set of degree-of-freedom values that
      // have the operations that are performed with them tracked.
      std::vector<ADNumberType> local_dof_values_ad (n_independent_variables);
      for (unsigned int i=0; i<n_independent_variables; ++i)
        local_dof_values_ad[i] = ADNumberType(n_independent_variables, i, local_dof_values[i]);

      // Compute all values, gradients etc. based on sensitive
      // AD degree-of-freedom values.
      scratch.fe_values_ref[u_fe].get_function_gradients_from_local_dof_values(
        local_dof_values_ad,
        scratch.solution_grads_u_total);

      // Accumulate the residual value for each degree of freedom.
      // Note: Its important that the vectors is initialised (zero'd) correctly.
      std::vector<ADNumberType> residual_ad (dofs_per_cell, ADNumberType(0.0));
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const Tensor<2,dim,ADNumberType> &grad_u = scratch.solution_grads_u_total[q_point];
          const Tensor<2,dim,ADNumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
          const ADNumberType               det_F = determinant(F);
          const Tensor<2,dim,ADNumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2,dim,ADNumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
          const Tensor<2,dim,ADNumberType> F_inv = invert(F);
          Assert(det_F > ADNumberType(0.0), ExcInternalError());

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const unsigned int k_group = fe.system_to_base_index(k).first.first;

              if (k_group == u_dof)
                {
                  scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                  scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
              else
                Assert(k_group <= u_dof, ExcInternalError());
            }

          const SymmetricTensor<2,dim,ADNumberType> tau = lqph[q_point]->get_tau(det_F,b_bar);

          // Next we define some position-dependent aliases, again to
          // make the assembly process easier to follow.
          const std::vector<SymmetricTensor<2, dim,ADNumberType> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
          const double JxW = scratch.fe_values_ref.JxW(q_point);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int i_group     = fe.system_to_base_index(i).first.first;

              if (i_group == u_dof)
                residual_ad[i] += (symm_grad_Nx[i] * tau) * JxW;
              else
                Assert(i_group <= u_dof, ExcInternalError());
            }
        }

      for (unsigned int I=0; I<n_independent_variables; ++I)
        {
          const ADNumberType &residual_I = residual_ad[I];
          data.cell_rhs(I) = -residual_I.val(); // RHS = - residual
          for (unsigned int J=0; J<n_independent_variables; ++J)
            {
              // Compute the gradients of the residual entry [forward-mode]
              data.cell_matrix(I,J) = residual_I.dx(J); // linearisation_IJ
            }
        }

    //  std::cout<<"ok"<<std::endl;
    }

  };


  template <int dim>
  struct Assembler<dim,Sacado::Rad::ADvar<Sacado::Fad::DFad<double> > > : Assembler_Base<dim,Sacado::Rad::ADvar<Sacado::Fad::DFad<double> > >
  {
    typedef Sacado::Fad::DFad<double>       ADDerivType;
    typedef Sacado::Rad::ADvar<ADDerivType> ADNumberType;
    using typename Assembler_Base<dim,ADNumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim,ADNumberType>::PerTaskData_ASM;

    virtual ~Assembler() {}

    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM &scratch,
                                              PerTaskData_ASM &data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int &n_q_points = data.solid->n_q_points;
      const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;

      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      const std::vector<std::shared_ptr<const PointHistory<dim,ADNumberType> > > lqph =
        data.solid->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      const unsigned int n_independent_variables = data.local_dof_indices.size();
      std::vector<double> local_dof_values(n_independent_variables);
      cell->get_dof_values(scratch.solution_total,
                           local_dof_values.begin(),
                           local_dof_values.end());

      // We now retrieve a set of degree-of-freedom values that
      // have the operations that are performed with them tracked.
      std::vector<ADNumberType> local_dof_values_ad (n_independent_variables);
      for (unsigned int i=0; i<n_independent_variables; ++i)
        local_dof_values_ad[i] = ADDerivType(n_independent_variables, i, local_dof_values[i]);

      // Compute all values, gradients etc. based on sensitive
      // AD degree-of-freedom values.
      scratch.fe_values_ref[u_fe].get_function_gradients_from_local_dof_values(
        local_dof_values_ad,
        scratch.solution_grads_u_total);

      // Next we compute the total potential energy of the element.
      // This is defined as follows:
      // Total energy = (internal - external) energies
      // Note: Its important that this value is initialised (zero'd) correctly.
      ADNumberType cell_energy_ad = ADNumberType(0.0);
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const Tensor<2,dim,ADNumberType> &grad_u = scratch.solution_grads_u_total[q_point];
          const Tensor<2,dim,ADNumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
          const ADNumberType               det_F = determinant(F);
          const Tensor<2,dim,ADNumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2,dim,ADNumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
          Assert(det_F > ADNumberType(0.0), ExcInternalError());

          // Next we define some position-dependent aliases, again to
          // make the assembly process easier to follow.
          const double JxW = scratch.fe_values_ref.JxW(q_point);

          const ADNumberType Psi = lqph[q_point]->get_Psi(det_F,b_bar);

          // We extract the configuration-dependent material energy
          // from our QPH history objects for the current quadrature point
          // and integrate its contribution to increment the total
          // cell energy.
          cell_energy_ad += Psi * JxW;
        }

      // Compute derivatives of reverse-mode AD variables
      ADNumberType::Gradcomp();

      for (unsigned int I=0; I<n_independent_variables; ++I)
        {
          // This computes the adjoint df/dX_{i} [reverse-mode]
          const ADDerivType residual_I = local_dof_values_ad[I].adj();
          data.cell_rhs(I) = -residual_I.val(); // RHS = - residual
          for (unsigned int J=0; J<n_independent_variables; ++J)
            {
              // Compute the gradients of the residual entry [forward-mode]
              data.cell_matrix(I,J) = residual_I.dx(J); // linearisation_IJ
            }
        }
    }

  };


#endif


// Since we use TBB for assembly, we simply setup a copy of the
// data structures required for the process and pass them, along
// with the memory addresses of the assembly functions to the
// WorkStream object for processing. Note that we must ensure that
// the matrix is reset before any assembly operations can occur.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::assemble_system(const BlockVector<double> &solution_delta)
  {

   // std::cout<<"Ok, we test the code"<<std::endl;	  
    timer.enter_subsection("Assemble linear system");
    std::cout << " ASM " << std::flush;

    tangent_matrix = 0.0;
    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_JxW_values);

    const BlockVector<double> solution_total(get_total_solution(solution_delta));
    static  typename Assembler_Base<dim,NumberType>::PerTaskData_ASM per_task_data(this);
   // per_task_data(this);

    typename Assembler_Base<dim,NumberType>::ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face, solution_total);
    Assembler<dim,NumberType> assembler;

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    static_cast<Assembler_Base<dim,NumberType>&>(assembler),
                    &Assembler_Base<dim,NumberType>::assemble_system_one_cell,
                    &Assembler_Base<dim,NumberType>::copy_local_to_global_ASM,
                    scratch_data,
                    per_task_data);

		  
     /* std::cout<<"########################show dev_cell _matrix#################################"<<std::endl;
      for (unsigned int i = 0; i < dofs_per_cell; ++i){
          for (unsigned int j = 0; j < dofs_per_cell; ++j){
          if(j!=dofs_per_cell){
             std::cout<<per_task_data.dev_cell_matrix(i,j)<<" ";

           }
        }
        std::cout<<std::endl;
   }
      std::cout<<"############################show dev_cell_matrix######################################"<<std::endl;
  */
//    std::cout<<"rows:"<<tangent_matrix.n_block_rows()<<std::endl;
  //  std::cout<<"cols"<<tangent_matrix.n_block_cols()<<std::endl;


 //  const typename BlockSparseMatrix<double>::value_type &block = tangent_matrix.block(0, 0);
/*
  for (unsigned int row = 0; row < block.m(); ++row) {
    for (unsigned int col = 0; col < block.n(); ++col) {
      std::cout << block(row, col) << " ";
    }
    std::cout << std::endl;
  }
*/




/*const unsigned int block_size = tangent_matrix.block(0, 0).m();
    
    for (unsigned int row = 0; row < block_size; ++row) {
      for (unsigned int col = 0; col < block_size; ++col) {
        const double value = tangent_matrix.block(0, 0)(row, col);
        
      }
    }
*/



 //  unsigned int num_blocks_rows = tangent_matrix.m();
  // unsigned int num_blocks_cols = tangent_matrix.n();

   //unsigned int blocks_rows = tangent_matrix.row();
  // unsigned int blocks_cols = tangent_matrix.column();



//unsigned int n_blocks_rows = tangent_matrix.n_block_rows();
//unsigned int n_blocks_cols = tangent_matrix.n_block_cols();


  // std::cout<<"rows of block in tangent_matrix:"<<num_blocks_rows<<std::endl;
  // std::cout<<"cols of block in tangent_matrix:"<<num_blocks_cols<<std::endl;



  // std::cout<<"rows of block in tangent_matrix:"<<blocks_rows<<std::endl;
  // std::cout<<"cols of block in tangent_matrix:"<<blocks_cols<<std::endl;


  // std::cout<<"rows of block in tangent_matrix:"<<n_blocks_rows<<std::endl;
  // std::cout<<"cols of block in tangent_matrix:"<<n_blocks_cols<<std::endl;
   
   // const SparseMatrix<double>& block_matrix = tangent_matrix.block(0, 0);
//    const BlockSparseMatrix<double>& block_matrix = tangent_matrix;
   // std::cout<<"show the start value of matrix"<<block_matrix(0,0)<<std::endl;
    //std::cout<<"show the end value of matrix"<<block_matrix(block_matrix.m()-1,block_matrix.n()-1)<<std::endl;
    //std::cout<<"show the start value of matrix"<<block_matrix.block(0,0)<<std::endl;
   // std::cout<<"show the end value of matrix"<<block_matrix(block_matrix.m()-1,block_matrix.n()-1)<<std::endl;
    
  //  unsigned int n_blocks_rows = tangent_matrix.n_block_rows();

    //std::cout<<"The number of block of tangent_matrix is :"<<n_blocks_rows<<std::endl;
//unsigned int n_blocks_cols = tangent_matrix.n_block_cols();
//
//
//
//
/*
    #pragma omp crtical 
    {
     
      int tid = omp_get_thread_num();
      std::string prefix="stiffnessMatrix_";
      std::string endx=".txt";
      std::string file_name=prefix+std::to_string(tid)+endx;
      std::ofstream file(file_name);
      
      if (!file) {
        std::cerr << "Open falied for " << "stiffnessMatrix.txt" << std::endl;
        return;
    }

      for (int i = 0; i <tangent_matrix.m() ; ++i) {
        for (int j = 0; j < tangent_matrix.n(); ++j) {
            file << block_matrix(i,j) << " ";
        }
        file << std::endl;
    }

       file.close();
    }
    */
    timer.leave_subsection();
  }


// @sect4{Solid::make_constraints}
// The constraints for this problem are simple to describe.
// However, since we are dealing with an iterative Newton method,
// it should be noted that any displacement constraints should only
// be specified at the zeroth iteration and subsequently no
// additional contributions are to be made since the constraints
// are already exactly satisfied.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::make_constraints(const int &it_nr)
  {
    std::cout << " CST " << std::flush;

    // Since the constraints are different at different Newton iterations, we
    // need to clear the constraints matrix and completely rebuild
    // it. However, after the first iteration, the constraints remain the same
    // and we can simply skip the rebuilding step if we do not clear it.
    if (it_nr > 1)
      return;
    const bool apply_dirichlet_bc = (it_nr == 0);

    // The boundary conditions for the indentation problem are as follows: On
    // the -x face (ID = 1), we set up a zero-displacement condition, -y and +y traction 
    // free boundary condition (don't need to take care); -z and +z faces (ID = 2) are 
    // not allowed to move along z axis so that it is a plane strain problem. 
    // Finally, as described earlier, +x face (ID = 11) has an the applied 
    // distributed shear force (converted by total force per unit area) which 
    // needs to be taken care as an inhomogeneous Newmann boundary condition.
    //
    // In the following, we will have to tell the function interpolation
    // boundary values which components of the solution vector should be
    // constrained (i.e., whether it's the x-, y-, z-displacements or
    // combinations thereof). This is done using ComponentMask objects (see
    // @ref GlossComponentMask) which we can get from the finite element if we
    // provide it with an extractor object for the component we wish to
    // select. To this end we first set up such extractor objects and later
    // use it when generating the relevant component masks:

    if (apply_dirichlet_bc)
    {
      constraints.clear();

      // Fixed left hand side of the beam
      {
        const int boundary_id = 1;
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                boundary_id,
                                                Functions::ZeroFunction<dim>(n_components),
                                                constraints,
                                                fe.component_mask(u_fe));
      }

      // Zero Z-displacement through thickness direction
      // This corresponds to a plane strain condition being imposed on the beam
      if (dim == 3)
      {
        const int boundary_id = 2;
        const FEValuesExtractors::Scalar z_displacement(2);
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                boundary_id,
                                                Functions::ZeroFunction<dim>(n_components),
                                                constraints,
                                                fe.component_mask(z_displacement));
      }
    }
    else
    {
      if (constraints.has_inhomogeneities())
      {
        AffineConstraints<double> homogeneous_constraints(constraints);
        for (unsigned int dof = 0; dof != dof_handler_ref.n_dofs(); ++dof)
          if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
            homogeneous_constraints.set_inhomogeneity(dof, 0.0);
        constraints.clear();
        constraints.copy_from(homogeneous_constraints);
      }
    }

    constraints.close();
  }

// @sect4{Solid::solve_linear_system}
// As the system is composed of a single block, defining a solution scheme
// for the linear problem is straight-forward.
  template <int dim,typename NumberType>
  std::pair<unsigned int, double>
  Solid<dim,NumberType>::solve_linear_system(BlockVector<double> &newton_update)
  {
    BlockVector<double> A(dofs_per_block);
    BlockVector<double> B(dofs_per_block);

    unsigned int lin_it = 0;
    double lin_res = 0.0;

    // We solve for the incremental displacement $d\mathbf{u}$.
    {
      timer.enter_subsection("Linear solver");
      std::cout << " SLV " << std::flush;
      if (parameters.type_lin == "CG")
        {
          const int solver_its = static_cast<unsigned int>(
                                    tangent_matrix.block(u_dof, u_dof).m()
                                    * parameters.max_iterations_lin);
          const double tol_sol = parameters.tol_lin
                                 * system_rhs.block(u_dof).l2_norm();

          SolverControl solver_control(solver_its, tol_sol);

          GrowingVectorMemory<Vector<double> > GVM;
          SolverCG<Vector<double> > solver_CG(solver_control, GVM);

          // We've chosen by default a SSOR preconditioner as it appears to
          // provide the fastest solver convergence characteristics for this
          // problem on a single-thread machine.  However, for multicore
          // computing, the Jacobi preconditioner which is multithreaded may
          // converge quicker for larger linear systems.
          PreconditionSelector<SparseMatrix<double>, Vector<double> >
          preconditioner (parameters.preconditioner_type,
                          parameters.preconditioner_relaxation);
          preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

          solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
                          newton_update.block(u_dof),
                          system_rhs.block(u_dof),
                          preconditioner);

          lin_it = solver_control.last_step();
          lin_res = solver_control.last_value();
        }
      else if (parameters.type_lin == "Direct")
        {
          // Otherwise if the problem is small
          // enough, a direct solver can be
          // utilised.
          SparseDirectUMFPACK A_direct;
          A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
          A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

          lin_it = 1;
          lin_res = 0.0;
        }
      else
        Assert (false, ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();
    }

    // Now that we have the displacement update, distribute the constraints
    // back to the Newton update:
    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
  }

// @sect4{Solid::output_results}
// Here we present how the results are written to file to be viewed
// using ParaView or Visit. The method is similar to that shown in the
// tutorials so will not be discussed in detail.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::output_results() const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Since we are dealing with a large deformation problem, it would be nice
    // to display the result on a displaced grid!  The MappingQEulerian class
    // linked with the DataOut class provides an interface through which this
    // can be achieved without physically moving the grid points in the
    // Triangulation object ourselves.  We first need to copy the solution to
    // a temporary vector and then create the Eulerian mapping. We also
    // specify the polynomial degree to the DataOut object in order to produce
    // a more refined output data set when higher order polynomials are used.
    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);

    std::ostringstream filename;
    filename << "solution-" << time.get_timestep() << "_test.vtk";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
  }

}


// @sect3{Main function}
// Lastly we provide the main driver function which appears
// no different to the other tutorials.





double Squared(double x) { return x*x; }

struct Problem {
	
	int n, m;
	std::vector<double> x0, xmin, xmax;
	double x_density_percell;
	double *cell_array;
	double *solver_vec;
	double *diff_solver_vec;
	int dofs_all_num;
	int dofs_per_cell;
	int tol_dimen;
	double norm_value;
	double x_density_coff;
	double norm2=0.0;
	int num_cells=0;
	double obj_c;
	double *dc_vector;
	double *dv_vector;
	Eigen::MatrixXd dense_matrix ;
        Eigen::MatrixXd inv_dense_matrix ;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver;
        Eigen::MatrixXd  dev_cell_matrix;
	Problem()
		: n(1)
		, m(2)
		, x0(1,0.1e6)
		, xmin(1,0.1e6)
		, xmax(1,1e6)
	{ 

	
	}

  void Obj(double *x, double *f0x, double *fx) {

    // void ObjSens(double *x, double *f0x, double *fx, double *df0dx, double *dfdx){
           // f0x[0] = 0;
            using namespace dealii;
            using namespace Cook_Membrane;
            using namespace std;
	    const unsigned int dim = 3;
            //df0dx()
        //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
/*

        std::string filename = "parameters.prm";
        std::ifstream file(filename);

       if (!file) {
          std::cerr << "Failed to open file: " << filename << std::endl;
          
       }

      std::string line;
      std::ofstream temp("temp.prm"); // 创建一个临时文件
      std::stringstream ss;
      ss << x[0];

      std::string shear_modulus = ss.str();
      size_t pos = shear_modulus.find('.');
      if (pos != std::string::npos) {
        shear_modulus.erase(shear_modulus.find_last_not_of('0') + 1);
        shear_modulus.erase(shear_modulus.find_last_not_of('.') + 1);
      }


     while (std::getline(file, line)) {
         if (line.find("set Shear modulus") != std::string::npos) {
            line = "  set Shear modulus = "+shear_modulus;
         }
         temp << line << std::endl; // 将每一行写入临时文件
      }

      file.close();
      temp.close();
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  */  
    std::string prm_file="parameters.prm";
 //   std::string prm_file1="parameters1.prm";
    //std::string prm_file="temp.prm";
         {
          deallog.depth_console(0);

          Parameters::AllParameters parameters(prm_file);
	  parameters.mu=x[0];
//	  Parameters::AllParameters parameters1(prm_file1);
//	  parameters1.mu=x[0]*1.000002;
	  x_density_percell=parameters.mu;
          if (parameters.automatic_differentiation_order == 0)
         {
          std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

          // Allow multi-threading
         // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
           //                                                   dealii::numbers::invalid_unsigned_int);

          typedef double NumberType;
          Solid<dim,NumberType> solid_3d(parameters);
          solid_3d.run();

           

             dofs_all_num=solid_3d.tangent_matrix.m();
	     int active_cells=solid_3d.triangulation.n_active_cells();
	     n=active_cells;
	     cell_array=new double[active_cells];
	     #pragma omp parallel for
	     for(int i=0;i<n;i++){
	  	   cell_array[i]=x_density_percell;
	    }


/*
	  typedef double NumberType;
          Solid<dim,NumberType> solid_3d_diff(parameters1);
          solid_3d_diff.run();
 */                  

	  solver_vec=new double[dofs_all_num];
          diff_solver_vec=new  double[dofs_all_num];
	  //df0dx=new double[dofs_all_num];
          //dfdx=new double [dofs_all_num];
          dofs_per_cell=solid_3d.dofs_per_cell;
	  std::cout<<"dofs:"<<dofs_per_cell<<std::endl;
         // Eigen::SparseMatrix<double> eigen_tangent_matrix(solid_3d.tangent_matrix.m(), solid_3d.tangent_matrix.n());
	  
	  
	  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
          Eigen::MatrixXd  problem_cell_matrix(solid_3d.dofs_per_cell,solid_3d.dofs_per_cell);
	  std::ifstream inputFile("cell_matrix_output.txt");

/*
	  for (unsigned int i = 0; i < dofs_per_cell; ++i) {
             for (unsigned int j = 0; j < dofs_per_cell; ++j) {
	       char value;
               inputFile >>std::noskipws>>value;
		       //problem_dev_cell_matrix(i, j);
	     if (value != ' ') {
                // 将字符转换为相应的数据类型（例如 double），并存储到 new_dev_cell_matrix 中
                 problem_dev_cell_matrix(i,j) = static_cast<double>(value - '0'); // 假设数据类型是 double
              }
             }
          }


*/

       for (int i = 0; i < dofs_per_cell; ++i) {
           for (int j = 0; j < dofs_per_cell; ++j) {
             double value;
             inputFile >> value; // 从文件中读取一个数值

            // 在读取数据后，继续读取字符，直到遇到空格符号为止
              char nextChar;
              while (inputFile.get(nextChar) && nextChar != ' ') {}

              problem_cell_matrix(i, j) = value;
          }
       }
      inputFile.close();

//	  dev_cell_matrix=problem_dev_cell_matrix;
          //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
/*
          #pragma omp parallel for
          for (unsigned int i = 0; i < solid_3d.tangent_matrix.n(); ++i) {
            for (dealii::BlockSparseMatrix<double>::const_iterator it = solid_3d.tangent_matrix.begin(i);
               it !=solid_3d.tangent_matrix.end(i); ++it) {
              eigen_tangent_matrix.insert(it->row(), it->column()) = it->value();
            }
          }
 */
//
//	    dense_matrix = eigen_tangent_matrix.toDense();
  //          inv_dense_matrix = dense_matrix.inverse();
//	    eigensolver.compute(dense_matrix);
           // #pragma omp parallel for
           /*	
           for(int i=0;i<dofs_all_num;i++){
                        solver_vec[i]=solid_3d.solution_n[i]; 
			f0x[0]+=abs(solid_3d.solution_n[i]);
			//diff_solver_vec[i]=solid_3d_diff.solution_n[i];
			//norm2+=std::pow(solid_3d.solution_n[i],2);
            }
             */
	   // norm2=std::sqrt(norm2);
           #pragma omp parallel for
            for(int j=0;j<n;j++){
			if(fx[0]<=0){
			fx[0]+=cell_array[j]-0.5*1e6;
			}
	     }

           //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    // Initialize the index and k
        std::vector<int> iH, jH;
        //std::vector<double> sH;
        int rmin=2;
	int nelx=solid_3d.nelx;
	int nely=solid_3d.nely;
	int nelz=solid_3d.nelz;
	int nele=nelx*nely*nelz;
	tol_dimen=nele;
	int iH_size = nele * (2 * (std::ceil(rmin) - 1) + 1) * (2 * (std::ceil(rmin) - 1) + 1);
        int k = 0;


    // Loop over all nodes to calculate the filter matrix
     /*    for (int k1 = 1; k1 <= nelz; ++k1)
       {
         for (int i1 = 1; i1 <= nelx; ++i1)
         {
            for (int j1 = 1; j1 <= nely; ++j1)
            {
                int e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1;
                for (int k2 = std::max(k1 - (static_cast<int>(std::ceil(rmin)) - 1), 1); k2 <= std::min(k1 + (static_cast<int>(std::ceil(rmin)) - 1), nelz); ++k2)
                {
                    for (int i2 = std::max(i1 - (static_cast<int>(std::ceil(rmin)) - 1), 1); i2 <= std::min(i1 + (static_cast<int>(std::ceil(rmin)) - 1), nelx); ++i2)
                    {
                        for (int j2 = std::max(j1 - (static_cast<int>(std::ceil(rmin)) - 1), 1); j2 <= std::min(j1 + (static_cast<int>(std::ceil(rmin)) - 1), nely); ++j2)
                        {
                            int e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2;
                            k = k + 1;
                            iH.push_back(e1 - 1);
                            jH.push_back(e2 - 1);
                            double dist = std::sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2));
                            sH.push_back(std::max(0.0, rmin - dist));
                        }
                    }
                }
            }
        }
    }

*/

//>>>>>>>>>>>>>>>>>>>>>
std::vector<std::vector<int>> nodegrd(nely + 1, std::vector<int>(nelx + 1));

    int counter = 1;
    #pragma omp for
    for (int i = 0; i <= nelx; i++) {
        for (int j = 0; j <= nely; j++) {
            nodegrd[j][i] = counter;
            counter++;
        }
     }
       

   std::vector<int> nodeids(nelx * nely);
   Eigen::MatrixXi eigen_nodeids(nely * nelx, 1);
    int index = 0;
    #pragma omp for
    for (int i = 0; i < nelx; i++) {
        for (int j = 0; j < nely; j++) {
            nodeids[index] = nodegrd[j][i];
            eigen_nodeids(index)=nodeids[index];
            index++;
        }
    }
    
    //128x1



    std::vector<int> nodeidz;
   // Eigen::MatrixXi eigen_nodeidz;
    int step = (nely + 1) * (nelx + 1);
   // int cnum=0;
    for (int i = 0; i <= (nelz - 1) * step; i += step) {
        nodeidz.push_back(i);
//	eigen_nodeidz(cnum)=i;
//	cnum++;
    }//
  
Eigen::MatrixXi eigen_nodeidz(nodeidz.size(),1);
//std::cout<<"--------------------"<<std::endl;
   #pragma omp for 
   for (int i = 0; i < nodeidz.size(); i++) {
       eigen_nodeidz(i)=nodeidz[i];
    //   std::cout<<eigen_nodeidz(i)<<" ";//8x1
    }
  //  std::cout<<std::endl;

//Eigen::MatrixXi eigen_nodeids(nely * nelx, 1);

//for(int i=0;i<nodeids.size();i++){
//eigen_nodeids(i)=nodeids[i];
//}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


std::vector<std::vector<std::vector<double>>> xPhys(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz)));
std::cout<<"-------------------mu is:"<<x_density_percell<<"------------------"<<std::endl;
  #pragma omp for
  for (int i = 0; i < nely; ++i) {
        for (int j = 0; j < nelx; ++j) {
            for (int k = 0; k < nelz; ++k) {
              xPhys[i][j][k]=x_density_percell;
            }
        }
    }




//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
int eigen_nodeidz_cols = eigen_nodeidz.transpose().cols();
    Eigen::MatrixXi replicated_nodeids = eigen_nodeids.replicate(1,eigen_nodeidz_cols);

    // Column replication of nodeidz
    int eigen_nodeids_rows = eigen_nodeids.rows();
    Eigen::MatrixXi replicated_nodeidz = eigen_nodeidz.transpose().replicate(eigen_nodeids_rows,1);
    // Element-wise addition of replicated matrices
   Eigen::MatrixXi result = replicated_nodeids + replicated_nodeidz;

   Eigen::Map<Eigen::VectorXi>result_vec(result.data(), result.size());


   Eigen::VectorXi edofVec(result_vec.rows());
   #pragma omp for
   for(int i=0;i<result_vec.rows();i++){

      edofVec(i) = result_vec(i)*3+1;
      //result_vec(i) = edofVec(i);
   }
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   Eigen::MatrixXi edofMat1(nele,dofs_per_cell);
   edofMat1 = edofVec.replicate(1, dofs_per_cell);
   //std::cout<<edofMat1<<std::endl;
    Eigen::VectorXd ele1(3);
    ele1 << 0, 1, 2;

    Eigen::VectorXd ele2(6);
    ele2 << 3, 4, 5, 0, 1, 2;
    ele2 = ele2.array() + 3 * nely; // Add 3*nely to each element of ele2

    Eigen::VectorXd ele3(3);
    ele3 << -3, -2, -1;

    // Concatenate ele1, ele2, and ele3 into test_ele
    Eigen::VectorXd test_ele(ele1.size() + ele2.size() + ele3.size());
    test_ele.block(0, 0, ele1.size(), 1) = ele1;
    test_ele.block(ele1.size(), 0, ele2.size(), 1) = ele2;
    test_ele.block(ele1.size() + ele2.size(), 0, ele3.size(), 1) = ele3;


    Eigen::VectorXd test_ele1(test_ele.size()*2);
    Eigen::VectorXd temp_ele(test_ele.size());

    temp_ele=3*(nely+1)*(nelx+1)+test_ele.array();

   // cout<<"-------------------"<<endl;
    test_ele1.block(0,0,test_ele.size(),1)=test_ele;
    test_ele1.block(test_ele.size(),0, temp_ele.size(),1)=temp_ele;
    //std::cout << "test_ele: \n" << test_ele1.transpose() << std::endl;
    Eigen::MatrixXi edofMat2(nele,dofs_per_cell);
    Eigen::VectorXi edofVec2(test_ele1.rows());
   
    edofVec2=test_ele1.cast<int>();
    edofMat2=edofVec2.transpose().replicate(nele,1);
   // std::cout<<edofMat2<<std::endl;
   // std::cout<<"-----------------------------------"<<std::endl;

    Eigen::MatrixXi edofMat=edofMat1+edofMat2 ;



    //std::cout<<edofMat<<std::endl;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  std::set<std::pair<int, int>> added_indices;
  //Eigen::SparseMatrix<double> H(nele, nele);
  std::vector<Eigen::Triplet<double>> sH;
  for (int k1 = 1; k1 <= nelz; ++k1) {
        for (int i1 = 1; i1 <= nelx; ++i1) {
            for (int j1 = 1; j1 <= nely; ++j1) {
                int e1 = (k1 - 1) * nelx * nely + (i1 - 1) * nely + j1;
                for (int k2 = std::max(k1 - static_cast<int>(std::ceil(rmin) - 1), 1); k2 <= std::min(k1 + static_cast<int>(std::ceil(rmin) - 1), nelz); ++k2) {
                    for (int i2 = std::max(i1 - static_cast<int>(std::ceil(rmin) - 1), 1); i2 <= std::min(i1 + static_cast<int>(std::ceil(rmin) - 1), nelx); ++i2) {
                        for (int j2 = std::max(j1 - static_cast<int>(std::ceil(rmin) - 1), 1); j2 <= std::min(j1 + static_cast<int>(std::ceil(rmin) - 1), nely); ++j2) {
                            int e2 = (k2 - 1) * nelx * nely + (i2 - 1) * nely + j2;

                            // Check if (e1, e2) pair is already added
                            if (added_indices.find(std::make_pair(e1, e2)) == added_indices.end()) {
                                iH.push_back(e1);
                                jH.push_back(e2);
                                //double distance_squared = (i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2);
                                //double distance = std::sqrt(distance_squared);
                                //sH.push_back(std::max(0.0, rmin - distance));
                                sH.push_back(Eigen::Triplet<double>(e1 - 1, e2 - 1, std::max(0.0, rmin - sqrt((i1 - i2) * (i1 - i2) + (j1 - j2) * (j1 - j2) + (k1 - k2) * (k1 - k2)))));

                                added_indices.insert(std::make_pair(e1, e2)); // Add the pair to the set
                            }
                        }
                    }
                }
            }
        }
    }
//<<<<<<<<<<<<<<<<<<<<<

auto maxih_element=std::max_element(iH.begin(),iH.end());
auto maxjh_element=std::max_element(jH.begin(),jH.end());
//std::cout<<"+++++++++++++++++iH max:"<<*maxih_element<<std::endl;
//std::cout<<"+++++++++++++++++jH max:"<<*maxjh_element<<std::endl;
Eigen::MatrixXd H_dense(*maxih_element,*maxjh_element);
Eigen::SparseMatrix<double> H(*maxih_element,*maxjh_element);
H.setFromTriplets(sH.begin(), sH.end());
Eigen::VectorXd Hs(*maxih_element);
//std::cout<<"-------------"<<*maxih_element<<","<<*maxjh_element<<std::endl;
//std::cout<<H.rows()<<","<<H.cols<<std::endl;
//std::cout<<Hs.rows()<<std::endl;
double sum_temp=0.0;
#pragma omp for
for(int i=0;i<*maxih_element;i++){
  for(int j=0;j<*maxih_element;j++){
     
     sum_temp+=H.coeff(i, j);
     H_dense(i,j)=H.coeff(i, j);
   }

   Hs[i]=sum_temp;
   sum_temp=0.0;

//   std::cout<<"############################("<<i<<",1): "<<Hs[i]<<std::endl;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 // const int n_dofs = dof_handler.n_dofs();active_cells
//    const int dofs_per_cell = fe.dofs_per_cell;dofs_per_cell
std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;

int num_cells=edofMat.rows();
int num_dofs=edofMat.cols();
Eigen::MatrixXd U_edofMat(num_cells, num_dofs);
#pragma omp for
for (int i = 0; i < num_cells; ++i) {
        for (int j = 0; j < num_dofs; ++j) {
            int dof_index = edofMat(i, j);  // Get dof index from edofMat
            if(dof_index<dofs_all_num){
	    U_edofMat(i, j) = solid_3d.solution_n[dof_index];
	    } // Extract corresponding U value
        }
    }
//std::cout << "U(edofMat):\n" << U_edofMat << std::endl;


Eigen::MatrixXd U_KE=U_edofMat*problem_cell_matrix;


Eigen::MatrixXd KE_U_multi_U=U_KE.array()*U_edofMat.array();

 Eigen::VectorXd sumVector =sumAlongRows(KE_U_multi_U);

// std::cout<<sumVector.rows()<<std::endl;


//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


dv_vector=new double[nely*nelx*nelz]; 
 
 
int penal=3;
double E0=1.0;
double Emin=1e-9;
std::vector<std::vector<std::vector<double>>> ce(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz)));
//std::vector<std::vector<std::vector<double>>> xPhys(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz)));
std::vector<std::vector<std::vector<double>>> c_temp(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz)));
std::vector<std::vector<std::vector<double>>> dc(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz)));
std::vector<std::vector<std::vector<double>>> dv(nely, std::vector<std::vector<double>>(nelx, std::vector<double>(nelz,1)));
    int idx = 0;
    #pragma omp for
    for (int i = 0; i < nely; ++i) {
        for (int j = 0; j < nelx; ++j) {
            for (int k = 0; k < nelz; ++k) {
                ce[i][j][k] =sumVector[idx];
		//xPhys[i][j][k]=1.3;
		dc[i][j][k]=-1*penal*(E0-Emin)*pow(xPhys[i][j][k],penal-1)*ce[i][j][k];
		xPhys[i][j][k]=Emin+pow(xPhys[i][j][k],penal-1)*(E0-Emin);
                c_temp[i][j][k]=ce[i][j][k]*xPhys[i][j][k];

                ++idx;
            }
        }
    }




  std::vector<std::vector<double>> sum_nelx_nelz(nelx, std::vector<double>(nelz, 0.0));



  for (int i = 0; i < nelx; ++i) {
      for (int j = 0; j < nelz; ++j) {
         for (int k = 0; k < nely; ++k) {
             sum_nelx_nelz[i][j] += c_temp[k][i][j];
         }
     }
  }


    std::vector<double> sum_nelz(nelz, 0.0);

    for (int i = 0; i < nelz; ++i) {
        for (int k = 0; k < nelx; ++k) {
           sum_nelz[i]=sum_nelx_nelz[k][i];
        }
    }



  double c=std::accumulate(sum_nelz.begin(),sum_nelz.end(),0.0);
  std::cout<<"-------****************************************************************************--------------------------"<<std::endl;
  std::cout<<c<<std::endl;
  std::cout<<"-------****************************************************************************--------------------------"<<std::endl;
         *f0x=c;
  //double obj_c;
// double *dc_vector;

  int dc_num=0;
  Eigen::VectorXd dc_vec(nely*nelx*nelz);
  Eigen::VectorXd dv_vec(nely*nelx*nelz);
 
   for (int k = 0; k < nelz; ++k) {
        for (int j = 0; j < nelx; ++j) {
            for (int i = 0; i < nely; ++i) {
               dc_vec(dc_num)=dc[i][j][k];
	       dv_vec(dc_num)=dv[i][j][k];
	       dc_num++;
            }
        }
    }
   
  dc_vector=new double[nely*nelx*nelz];
  dc_vec=dc_vec.array()/Hs.array();
  dc_vec=H_dense*dc_vec;

//  dv_vector=new double[nely*nelx*nelz];
  //x_density_percell


  #pragma omp for
  for(int i=0;i<dc_vec.rows();i++){
   dc_vector[i] = dc_vec[i];
   dv_vector[i] = dv_vec[i];
  }
/*
  for (int k = 0; k < nelz; ++k) {
        for (int j = 0; j < nelx; ++j) {
            for (int i = 0; i < nely; ++i) {
             //dv_vector[i][j][k]=1;

            }
        }
    }
*/
//std::cout<<dc_vec<<std::endl;

//std::cout<<"-----------------------end of obj--------------------------------------"<<std::endl;
//double E0=1.0;
//double Emin=1e-9;



//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//double xPhys=x[0];



	 //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   


         }
	 
        }
	

        /* for(int i=0;i<dofs_all_num;i++){
             if(solver_vec[i]>0){

                    df0dx[i]=1;
              }
           else if(solver_vec[i]<0){
                  df0dx[i]=-1;
             }
              else{
                   df0dx[i]=;

                    }

          }
	  */
//
  //       std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;

       // std::vector<double> x_diff = {x[0]+4};//?
        //std::cout<<"x_diff ->:"<<std::setprecision(14)<<x_diff[0]<<std::endl;
       // Obj_diff(x_diff.data(),diff_solver_vec);

     /*   for(int i=0;i<dofs_all_num;i++){
              if(solver_vec[i]>0){

                    df0dx[i]=1;
              }
             else if(solver_vec[i]<0){
                  df0dx[i]=-1;
             }
                 else{
                                 df0dx[i]=0;
           
                    }
            
          }//求出目标函数导数df0dx
    */
      // std::cout<<"-----------------------difference finite elements method-----------------------------------"<<std::endl;
      // for(int k=0;k<dofs_all_num;k++){

        //       dfdx[k]=(solver_vec[k]-diff_solver_vec[k])*0.25;
          //     std::cout<<"dfdx["<<k<<"]:"<<dfdx[k]<<std::endl;

      // }
//	 std::cout<<"--------------------Test end in function diff finite methodj---------------------------------"<<std::endl;
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
}


/*
   void Obj_diff(double *x,double *diff_solver_vec) {
      using namespace dealii;
      using namespace Cook_Membrane;
      using namespace std;
      const unsigned int dim = 3;

 
       std::string prm_file1="parameters1.prm";
         {
             deallog.depth_console(0);

              Parameters::AllParameters parameters1(prm_file1);
	      parameters1.mu=x[0];
	     // x_density_percell=parameters.mu;
          if (parameters1.automatic_differentiation_order == 0)
         {
            std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

          // Allow multi-threading
         // Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
           //                                                   dealii::numbers::invalid_unsigned_int);

         typedef double NumberType;
         Solid<dim,NumberType> solid_3d_diff(parameters1);
         solid_3d_diff.run();
            // int dofs_all_num=solid_3d_diff.tangent_matrix.m();
	    // int active_cells=solid_3d_diff.triangulation.n_active_cells();
	    // int n=active_cells;
	     
	       

	   // diff_solver_vec=new  double[dofs_all_num];
 
	    for(int i=0;i<dofs_all_num;i++){
            
			diff_solver_vec[i]=solid_3d_diff.solution_n[i]; 
		
			//norm2+=std::pow(solid_3d_diff.solution_n[i],2);
                 }
        

	     }
	 
        } 

	 std::cout<<"finish the obj_diff"<<std::endl;
      }


*/

  Eigen::VectorXd sumAlongRows(const Eigen::MatrixXd& matrix) {
    Eigen::VectorXd sumVector(matrix.rows());
    #pragma omp for
    for (int i = 0; i < matrix.rows(); ++i) {
        sumVector(i) = matrix.row(i).sum();
    }
    return sumVector;
  }








void ObjSens(double *x, double *f0x, double *fx, double *df0dx, double *dfdx) {
             Obj(x, f0x, fx);
            // std::cout<<tol_dimen<<std::endl;
	     //df0dx=new double[tol_dimen];
	   /*    for(int i=0;i<tol_dimen;i++){
	     df0dx[i]=dc_vector[i];
	     }
	     dfdx[0]=0;
          */
          // std::vector<double> modified_df(df0dx, df0dx + tol_dimen);

	  // for(int i=0;i<tol_dimen;i++){
        // modified_df.push_back(dc_vector[i]);
	  // }
          //std::cout<<"--------------------------------------------------------------------"<<std::endl; 
            
             #pragma omp for
	     for(int i=0;i<tol_dimen;i++){
               df0dx[i] = dc_vector[i];
               dfdx[i] = dv_vector[i];
	    //   std::cout<<df0dx[i]<<std::endl;
              // df0dx->push_back(dc_vector[i]);
           }





}








/*

void ObjSens(double *x, double *f0x, double *fx, double *df0dx, double *dfdx) {
	    Obj(x, f0x, fx);
	   // Eigen::VectorXd df0dx_vector(dofs_all_num);
	   // Eigen::VectorXd f0x_vector(dofs_all_num);
	  // double *temp_solver=new double[dofs_all_num];

	    for(int i=0;i<dofs_all_num;i++){
			if(solver_vec[i]>0){

			    //df0dx[i]=solver_vec[i]/norm2;
			    df0dx[i]=1;
			    //temp_solver[i]=solver_vec[i];
			    //df0dx_vector(i)=df0dx[i];
			    //f0x_vector(i)=solver_vec[i];
			}
			else if(solver_vec[i]<0){
			        df0dx[i]=-1;	
				//df0dx_vector(i)=df0dx[i];
				//f0x_vector(i)=solver_vec[i];
			//	temp_solver[i]=solver_vec[i];
			       }
			else{
				 df0dx[i]=0;
                                 //df0dx_vector(i)=df0dx[i];
				 //f0x_vector(i)=solver_vec[i];
			//	 temp_solver[i]=solver_vec[i];
			     }
		//	f0x_vector(i)=solver_vec[i];
		     }//求出目标函数导数df0dx
	//	double *temp_solver=new double[dofs_all_num];
	        	


        //double diff_scale=;
       // double f_diff=0.0;
       // std::vector<double> g_diff(2,0.0);
       // std::vector<double> x_diff = {1.00002*x[0]};//?
	//std::cout<<"x_diff ->:"<<std::setprecision(14)<<x_diff[0]<<std::endl;
       // Obj_diff(x_diff.data());

       std::cout<<"-----------------------difference finite elements method-----------------------------------"<<std::endl;
       for(int k=0;k<dofs_all_num;k++){
       
	       dfdx[k]=(solver_vec[k]-diff_solver_vec[k])/(1e-6);
               std::cout<<"dfdx["<<k<<"]:"<<dfdx[k]<<std::endl;

       } 

**/
			
//	dev_cell_matrix
   /*  std::cout<<"-------------------------------------------------------------------------------------"<<std::endl;
     for(int i=0;i<dofs_per_cell;i++){
	for(int j=0;j<dofs_per_cell;j++){
		std::cout<<dev_cell_matrix(i,j)<<" ";
	}
         std::cout<<std::endl;
     }
     std::cout<<"-------------------------------------------------------------------------------------"<<std::endl;
   */

		//>>>伴随法
     /*
	   Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
           Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
	   Eigen::MatrixXd B = eigenvectors;
           Eigen::MatrixXd D = Eigen::MatrixXd::Zero(B.cols(), B.cols());
 	   Eigen::VectorXd multi_vector(B.cols());
//	#pragma omp parallel for
        for (int i = 0; i < B.cols(); i++) {
          D(i, i) = std::sqrt(eigenvalues(i))*f0x_vector(i);
          multi_vector(i)=D(i, i);
        }
     */

	
  //     Eigen::VectorXd solver_vector(dofs_all_num);
		 
      // Eigen::VectorXd solver_gradient=-df0dx_vector*inv_dense_matrix*D*f0x_vector;
     //   Eigen::VectorXd solver_gradient(B.cols());
      /*
      solver_gradient=-df0dx_vector*inv_dense_matrix*multi_vector;
	   
	   for(int i=0;i<dofs_all_num;i++){
		   
		   dfdx[i]=solver_gradient(i);
		   
	   }//求出梯度
       */
  /*     //std::cout<<"-----------------obj sensitivity test---------------------"<<std::endl;
       
	}*/
	

};


void Print(double *x, int n, const std::string &name = "x") {
	std::cout << name << ":";
	for (int i=0;i<n;i++) {
		std::cout << " " << x[i];
	}
	std::cout << std::endl;
}


int main(int argc, char *argv[]) {
	
	using namespace dealii;
        using namespace Cook_Membrane;
//	using namespace Parameters;
        const unsigned int dim = 3;	
	std::cout << "///////////////////////////////////////////////////" << std::endl;
	std::cout << "// Test the GCMMA algorithm" << std::endl;
	std::cout << "///////////////////////////////////////////////////" << std::endl;

	/**{
	
        deallog.depth_console(0);
          Parameters::AllParameters parameters("parameters.prm");
	 if (parameters.automatic_differentiation_order == 0)
        {
          std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

          // Allow multi-threading
          Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                              dealii::numbers::invalid_unsigned_int);

          typedef double NumberType;
          Solid<dim,NumberType> solid_3d(parameters);
          solid_3d.run();
        }
	}
	**/
        //Problem toy;
 



	Problem toy;
	double movlim = 0.2;
/*
	std::vector<double> f=toy.xmin;
	std::vector<double> fnewi(1,0);
	std::vector<double> df(toy.n,0);//n=1
	std::vector<double> g(toy.m,0);
	std::vector<double> gnew(toy.m,0);//m=2,
	std::vector<double> dg(toy.n * toy.m,0);//n*m=2
        
	std::vector<double> x= toy.x0;//x0=0.002e6
	std::vector<double> xold=x;
//        xold=x;
	std::vector<double> xnew(toy.n,0);//n=1

	// Print initial values
	toy.Obj(x.data(), f.data(), g.data());
	//std::cout << "f: " << f << std::endl;
	//Print(g.data(), toy.m, "g");
*/

        int dofs_num=0;
	double f, fnew;
	//std::vector<double> df(toy.n);
	std::vector<double> g(2), gnew(2);
//	std::vector<double> dg(toy.n * toy.m);

	std::vector<double> x = toy.x0;//?
	std::vector<double> xold = x;
	std::vector<double> xnew(toy.n);
	toy.Obj(x.data(), &f, g.data());
         //std::vector<double> df;
	// double *df;
	//std::vector<double> df(1024,0.0);
	 //std::vector<double> df(toy.dofs_all_num);
        // std::vector<double> dg;
	// double *dg;
	//std::vector<double> df={0.0};
	//std::vector<double> dg(1024,0.0);
	//std::vector<double> dg={0.0};
	// std::vector<double> dg(toy.dofs_all_num);
         std::cout << "f: " << f << std::endl;
	 Print(g.data(), toy.m, "g");
        // toy.ObjSens(x.data(), &f, g.data(), df.data(), dg.data());
	std::vector<double> df(toy.tol_dimen,0.0);
	std::vector<double> dg(toy.tol_dimen,0.0);
	f=0.0;
	g[0]=0.0,g[1]=0.0;
	toy.ObjSens(x.data(), &f, g.data(), df.data(), dg.data());
        std::cout << "f: " << f << std::endl; 
 	Print(g.data(), toy.m, "g");
//        std::vector<double> y={x[0]*1.00002};
  //      toy.Obj_diff(y.data());


	// Initialize GCMMA
        GCMMASolver gcmma(toy.n, toy.m, 0, 1000, 1);
	MMASolver mma(toy.n, toy.m, 0, 1000, 1);
/*
	double ch = 1.0;
	int maxoutit = 8;
	for (int iter = 0; ch > 0.0002 && iter < maxoutit; ++iter) {
		toy.ObjSens(x.data(), &f, g.data(), df.data(), dg.data());

		// Call the update method
		if (0) {
			// MMA version
			mma.Update(x.data(), df.data(), g.data(), dg.data(),
				toy.xmin.data(), toy.xmax.data());
		} else {
			// GCMMA version
			gcmma.OuterUpdate(xnew.data(), x.data(), f, df.data(),
				g.data(), dg.data(), toy.xmin.data(), toy.xmax.data());

			// Check conservativity
			toy.Obj(xnew.data(), &fnew, gnew.data());
			bool conserv = gcmma.ConCheck(fnew, gnew.data());
			//std::cout << conserv << std::endl;
			for (int inneriter = 0; !conserv && inneriter < 15; ++inneriter) {
				// Inner iteration update
				gcmma.InnerUpdate(xnew.data(), fnew, gnew.data(), x.data(), f,
					df.data(), g.data(), dg.data(), toy.xmin.data(), toy.xmax.data());

				// Check conservativity
				toy.Obj(xnew.data(), &fnew, gnew.data());
				conserv = gcmma.ConCheck(fnew, gnew.data());
				//std::cout << conserv << std::endl;
			}
			x = xnew;
		}

		// Compute infnorm on design change
		ch = 0.0;
		for (int i=0; i < toy.n; ++i) {
			ch = std::max(ch, std::abs(x[i] - xold[i]));
			xold[i] = x[i];
		}

		// Print to screen
		printf("it.: %d, obj.: %f, ch.: %f \n", iter, f, ch);
		Print(x.data(), toy.n);
		toy.Obj(x.data(), &f, g.data());
		std::cout << "f: " << f << std::endl;
		Print(g.data(), toy.m, "g");
		std::cout << std::endl;
	}
*/
	return 0;
}




