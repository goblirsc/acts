// This file is part of the ACTS project.
//
// Copyright (C) 2016-2018 ACTS project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef ACTS_STRAIGHTLINE_STEPPER_HPP
#define ACTS_STRAIGHTLINE_STEPPER_HPP

#include <cmath>
#include "ACTS/EventData/TrackParameters.hpp"
#include "ACTS/MagneticField/concept/AnyFieldLookup.hpp"
#include "ACTS/Propagator/detail/ConstrainedStep.hpp"
#include "ACTS/Surfaces/Surface.hpp"
#include "ACTS/Utilities/Definitions.hpp"
#include "ACTS/Utilities/Units.hpp"

namespace Acts {

/// StraightLineStepper
///
/// The straight line stepper is a simple navigation stepper
/// to be used to navigate through the tracking geometry. It can be
/// used for simple material mapping, navigation validation
class StraightLineStepper
{

private:
  // This struct is a meta-function which normally maps to BoundParameters...
  template <typename T, typename S>
  struct s
  {
    typedef BoundParameters type;
  };

  // ...unless type S is int, in which case it maps to Curvilinear parameters
  template <typename T>
  struct s<T, int>
  {
    typedef CurvilinearParameters type;
  };

public:
  typedef detail::ConstrainedStep cstep;

  /// State for track parameter propagation
  ///
  struct State
  {
    /// Constructor from the initial track parameters
    /// @param [in] par The track parameters at start
    template <typename T>
    explicit State(const T&            par,
                   NavigationDirection ndir = forward,
                   double ssize = std::numeric_limits<double>::max())
      : pos(par.position())
      , dir(par.momentum().normalized())
      , qop(par.charge() / par.momentum().norm())
      , navDir(ndir)
      , accumulatedPath(0.)
      , stepSize(ndir * ssize)
    {
    }

    /// Global particle position accessor
    Vector3D
    position() const
    {
      return pos;
    }

    /// Momentum direction accessor
    Vector3D
    direction() const
    {
      return dir;
    }

    /// Actual momentum accessor
    Vector3D
    momentum() const
    {
      return (1. / qop) * dir;
    }

    /// Global particle position
    Vector3D pos = Vector3D(0, 0, 0);

    /// Momentum direction (normalized)
    Vector3D dir = Vector3D(1, 0, 0);

    /// Charge-momentum ratio, in natural units
    double qop = 1;

    /// Navigation direction, this is needed for searching
    NavigationDirection navDir;

    /// accummulated path length state
    double accumulatedPath = 0.;

    /// adaptive step size of the runge-kutta integration
    cstep stepSize = std::numeric_limits<double>::max();
  };

  /// Always use the same propagation state type, independently of the initial
  /// track parameter type and of the target surface
  template <typename T, typename S = int>
  using state_type = State;

  /// Intermediate track parameters are always in curvilinear parametrization
  template <typename T>
  using step_parameter_type = CurvilinearParameters;

  /// Return parameter types depend on the propagation mode:
  /// - when propagating to a surface we return BoundParameters
  /// - otherwise CurvilinearParameters
  template <typename T, typename S = int>
  using return_parameter_type = typename s<T, S>::type;

  /// Constructor
  StraightLineStepper() = default;

  /// Convert the propagation state (global) to curvilinear parameters
  /// @param state The stepper state
  /// @return curvilinear parameters
  static CurvilinearParameters
  convert(State& state)
  {
    double charge = state.qop > 0. ? 1. : -1.;
    // return the parameters
    return CurvilinearParameters(
        nullptr, state.pos, state.dir / std::abs(state.qop), charge);
  }

  /// Convert the propagation state to track parameters at a certain surface
  ///
  /// @tparam S The surface type
  ///
  /// @param [in] state Propagation state used
  /// @param [in] surface Destination surface to which the conversion is done
  ///
  /// @return are parameters bound to the target surface
  template <typename S>
  static BoundParameters
  convert(State& state, const S& surface)
  {
    double charge = state.qop > 0. ? 1. : -1.;
    // return the bound parameters
    return BoundParameters(
        nullptr, state.pos, state.dir / std::abs(state.qop), charge, surface);
  }

  /// Perform a straight line propagation step
  ///
  /// @param[in,out] state is the propagation state associated with the track
  ///                parameters that are being propagated.
  ///                The state contains the desired step size,
  ///                it can be negative during backwards track propagation,
  ///                and since we're using an adaptive algorithm, it can
  ///                be modified by the stepper class during propagation.
  ///
  /// @return the step size taken
  double
  step(State& state) const
  {
    // use the adjusted step size
    const double h = state.stepSize;
    // Update the track parameters according to the equations of motion
    state.pos += h * state.dir;
    // state the path length
    state.accumulatedPath += h;
    // return h
    return h;
  }
};

}  // namespace Acts

#endif  // ACTS_STRAIGHTLINE_STEPPER_HPP
