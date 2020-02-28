// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <functional>
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Vertexing/LinearizedTrack.hpp"

namespace Acts {

/// @class TrackAtVertex
///
/// @brief Defines a track at vertex object
///
/// @tparam input_track_t Track object type

template <typename input_track_t>

struct TrackAtVertex {
  /// Deleted default constructor
  TrackAtVertex() = delete;

  /// @brief Parameterized constructor
  ///
  /// @param chi2perTrack Chi2 of track
  /// @param paramsAtVertex Fitted perigee parameter
  /// @param originalParams Original perigee parameter
  TrackAtVertex(double chi2perTrack, const BoundParameters& paramsAtVertex,
                const input_track_t& originalParams)
      : chi2Track(chi2perTrack),
        ndf(0.),
        fittedParams(paramsAtVertex),
        originalTrack(originalParams),
        trackWeight(1.),
        vertexCompatibility(0.) {
  }

  /// @brief Constructor with default chi2
  ///
  /// @param chi2perTrack Chi2 of track
  /// @param paramsAtVertex Fitted perigee parameter
  /// @param originalParams Original perigee parameter
  TrackAtVertex(const BoundParameters& paramsAtVertex,
                const input_track_t& originalParams)
      : chi2Track(0.),
        ndf(0.),
        fittedParams(paramsAtVertex),
        originalTrack(originalParams),
        trackWeight(1.),
        vertexCompatibility(0.) {
  }

  /// Chi2 of track
  double chi2Track;

  /// Number degrees of freedom
  /// Note: Can be different from integer value
  /// since annealing can result in effective
  /// non-interger values
  double ndf;

  /// Fitted perigee
  BoundParameters fittedParams;

  /// Original input track
  // TODO: to be fully replaced by pointer version below
  input_track_t originalTrack;

  /// Original input parameters
  input_track_t* originalTrack2;

  /// Weight of track in fit
  double trackWeight;

  /// The linearized state of the track at vertex
  LinearizedTrack linearizedState;

  /// Value of the compatibility of the track to the actual vertex, based
  /// on the estimation of the 3d distance between the track and the vertex
  double vertexCompatibility;

};

}  // namespace Acts
