// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Mille/ActsToMille.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Alignment.hpp"
#include "ActsPlugins/Mille/Helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <numeric>
#include <set>

#include <Eigen/src/Core/Matrix.h>
#include <Mille/MilleDataStructures.h>

#include "Mille/MilleDecoder.h"

namespace ActsPlugins::ActsToMille {

namespace {

/// for the ACTS-internal indices, start counting at 0
unsigned long internalIndexSurfToParam(unsigned long surfaceIndex,
                                       unsigned long dofIndex) {
  return surfaceIndex * Acts::eAlignmentSize + dofIndex;
}
/// for global alignment parameters, start counting at 1 (fotran convention) for
/// Mille
unsigned long globalIndexSurfToParam(unsigned long surfaceIndex,
                                     unsigned long dofIndex) {
  return surfaceIndex * Acts::eAlignmentSize + dofIndex + 1;
}

}  // namespace

void dumpToMille(const ActsAlignment::detail::TrackAlignmentState& state,
                 MilleRecord& record) {
  // prepare the vectors to interface to Mille
  std::vector<int> globalIndices(state.alignmentDof, 0.);

  std::vector<double> localDeriv(state.trackParametersDim, 0.);
  std::vector<double> globalDeriv(state.alignmentDof, 0.);
  std::vector<unsigned int> localIndices(state.trackParametersDim, 0);
  // prepare the track parameter index array (always the same)
  std::iota(localIndices.begin(), localIndices.end(), 1);

  // map the alignment parameter labels.
  // Important: Millepede expects indices to start with 1
  std::vector<std::pair<int, int>> aliParLocalToGlobal;
  for (auto& [surf, indices] : state.alignedSurfaces) {
    auto& [globalSurfIndex, localStartIndex] = indices;
    for (std::size_t iPar = 0; iPar < Acts::eAlignmentSize; ++iPar) {
      aliParLocalToGlobal.emplace_back(
          internalIndexSurfToParam(localStartIndex, iPar),
          globalIndexSurfToParam(globalSurfIndex, iPar));
    }
  }

  /// Analyse the global cov a bit
  std::map<double, int> trkParByCov;
  for (std::size_t k = 0; k < state.trackParametersDim; ++k) {
    trkParByCov.emplace(state.trackParametersCovariance(k, k), k);
  }
  // std::cout << " ======== Sorted track parameter covariances: =======
  // "<<std::endl;
  std::vector<std::string> parTypeNames{"l0",    "l1",     "phi",
                                        "theta", "qoverp", "time"};
  std::set<std::size_t> kickUsOut = {};
  // std::cout << std::setw(8) << " Par: "<<"  "<<std::setw(12)<<" Cov "<<"  "<<
  // std::setw(7) <<" type "<<std::endl;
  double prev = 0;
  for (auto& [cov, index] : trkParByCov) {
    // std::cout << std::setw(8) << index <<"  "<<std::setw(12)<<cov<<"  "<<
    // std::setw(7) << parTypeNames[index % parTypeNames.size()]<< std::endl;
    if (prev != 0 && cov > 1e6 * prev) {
      std::cout << " Detected unconstrained parameter at index " << index
                << " - will suppress from alignment fit. " << std::endl;
      kickUsOut.insert(index);
    } else {
      prev = cov;
    }
  }
  // std::cout << " ----------------------------------------------------
  // "<<std::endl;

  std::size_t effectiveTrackParDim =
      state.trackParametersDim - kickUsOut.size();

  /// 1) write out the local measurements on the surfaces and their direct
  /// derivatives. This will populate the upper / left three quadrants of the
  /// alignment matrix, including direct correlations between alignment and
  /// track parameters.

  /// TODO: Add explicit diagonalisation for correlated (stereo) measurements.
  for (std::size_t iMeas = 0; iMeas < state.measurementDim; ++iMeas) {
    // arrange the global parameters correctly
    for (auto& [srcGlobal, destGlobal] : aliParLocalToGlobal) {
      // index for each global derivative
      globalIndices[srcGlobal] = destGlobal;
      // value for each global derivative
      globalDeriv[srcGlobal] =
          state.alignmentToResidualDerivative(iMeas, srcGlobal);
    }
    localDeriv.assign(localDeriv.size(), 0.);
    std::size_t iPar = 0;
    // local derivatives due to measurement uncertainties
    for (std::size_t iTrkPar = 0; iTrkPar < state.trackParametersDim;
         ++iTrkPar) {
      if (kickUsOut.contains(iTrkPar)) {
        continue;
      }
      localDeriv[iPar] = state.projectionMatrix(iMeas, iTrkPar);
      ++iPar;
    }
    // write a measurement to the ongoing Mille record.
    record.addData(
        // residual
        state.residual(iMeas),
        // sigma
        std::sqrt(state.measurementCovariance(iMeas, iMeas)),
        // local parameter indices
        localIndices,
        // local derivatives
        localDeriv, globalIndices, globalDeriv);
  }

  /// 2) Write out additional pseudo-measurements representing the (local) track
  /// parameter correlations from the Kalman fit (linearisation point).
  /// These enter the bottom right quadrant of the alignment matrix and
  /// represent the additional contributions to the track fit chi2
  /// arising from the correlations between the parameters on different
  /// surfaces encoded in the Kalman track model.
  /// Step 1) already added the terms arising from measurement uncertainties,
  /// now we need to extend this to the full Kalman covariance.
  /// This requires "unfitting" the tracks, making this a rather slow
  /// and numerically tricky step.

  /// compute the part of the weight matrix arising from Step 1).
  /// This is already present in the Mille record and should not
  /// be duplicated

  Acts::DynamicMatrix reducedCovariance{effectiveTrackParDim,
                                        effectiveTrackParDim};
  Acts::DynamicMatrix reducedProjection{state.measurementDim,
                                        effectiveTrackParDim};
  std::size_t ix = 0;
  for (std::size_t k = 0; k < state.trackParametersDim; ++k) {
    std::size_t iy = 0;
    if (kickUsOut.contains(k)) {
      continue;
    }
    for (std::size_t m = 0; m < state.measurementDim; ++m) {
      reducedProjection(m, ix) = state.projectionMatrix(m, k);
    }
    for (std::size_t l = 0; l < state.trackParametersDim; ++l) {
      if (kickUsOut.contains(l)) {
        continue;
      }
      reducedCovariance(ix, iy) = state.trackParametersCovariance(k, l);
      ++iy;
    }
    ++ix;
  }

  const Acts::DynamicMatrix weightMatMeasurements =
      state.projectionMatrix.transpose() *
      state.measurementCovariance.inverse() * state.projectionMatrix;

  const Acts::DynamicMatrix reducedWeightMatMeasurements =
      reducedProjection.transpose() * state.measurementCovariance.inverse() *
      reducedProjection;

  // regularise the (full) Kalman covariance. This is needed to stabilise
  // poorly constrained directions (usually: time)
  const Acts::DynamicMatrix regularisedCov =
      regulariseCovariance(state.trackParametersCovariance);

  const Acts::DynamicMatrix reducedRegularisedCov =
      // regulariseCovariance(reducedCovariance,1.e-15,5000.,0);
      regulariseCovariance(reducedCovariance, 1.e-12, 5000., 1e-15);

  // now we can get the piece of the weight matrix not already covered by
  // the measurement uncertainties
  const Acts::DynamicMatrix correlationTerm =
      getInverseComplement(regularisedCov, weightMatMeasurements);

  const Acts::DynamicMatrix reducedCorrelationTerm =
      getInverseComplement(reducedRegularisedCov, reducedWeightMatMeasurements);

  // std::cout << " Original covariance
  // ("<<state.trackParametersCovariance.rows()<<" x
  // "<<state.trackParametersCovariance.cols()<<")" <<std::endl <<
  // state.trackParametersCovariance.block(0,0,8,8)<< std::endl; std::cout << "
  // Reduced covariance ("<<reducedCovariance.rows()<<" x
  // "<<reducedCovariance.cols()<<")" <<std::endl <<
  // reducedCovariance.block(0,0,8,8)<< std::endl; std::cout << " Original
  // projection ("<<state.projectionMatrix.rows()<<" x
  // "<<state.projectionMatrix.cols()<<")" <<std::endl <<
  // state.projectionMatrix.block(0,0,8,8)<< std::endl; std::cout << " Reduced
  // projection ("<<reducedProjection.rows()<<" x
  // "<<reducedProjection.cols()<<")" <<std::endl <<
  // reducedProjection.block(0,0,8,8)<< std::endl; std::cout << " Original meas
  // weight mat ("<<weightMatMeasurements.rows()<<" x
  // "<<weightMatMeasurements.cols()<<")" <<std::endl <<
  // weightMatMeasurements.block(0,0,8,8)<< std::endl; std::cout << " Reduced
  // meas weight mat ("<<reducedWeightMatMeasurements.rows()<<" x
  // "<<reducedWeightMatMeasurements.cols()<<")" <<std::endl <<
  // reducedWeightMatMeasurements.block(0,0,8,8)<< std::endl; std::cout << "
  // Original correlation cov ("<<correlationTerm.rows()<<" x
  // "<<correlationTerm.cols()<<")" <<std::endl <<
  // correlationTerm.block(0,0,8,8)<< std::endl;

  // std::cout << " Alignment derivatives:
  // ("<<state.alignmentToResidualDerivative.rows()<<" x
  // "<<state.alignmentToResidualDerivative.cols()<<") : " <<std::endl; for (int
  // meas = 0; meas < 9; ++meas){
  //   for (int sen = 0; sen < 9; ++sen){
  //     std::cout <<"   Measurements on surface "<<meas<<" by DoF for surface
  //     "<<sen<<":"<<std::endl; std::cout  <<
  //     state.alignmentToResidualDerivative.block(meas * 2, sen * 6,2,6)<<
  //     std::endl;
  //   }
  // }

  // Decompose the matrix we need to add into a sum of rank-1 matrices,
  // C_add = sum (lambda_i v_i v_i^T), which can be interpreted
  // as pseudo-measurements with sigma_i 1/sqrt(lambda_i) and local derivatives
  // v_i. This relies on C_add being symmetric positive (semi)definite.
  Eigen::SelfAdjointEigenSolver<Acts::DynamicMatrix> eigenSolver(
      reducedCorrelationTerm);
  if (eigenSolver.info() != Eigen::Success) {
    std::cout << " FAILED to find decompose correlation term" << std::endl;
    return;
  }
  const Acts::DynamicVector eigenVals = eigenSolver.eigenvalues();
  const Acts::DynamicMatrix eigenVecs = eigenSolver.eigenvectors();
  //
  std::cout << " Eigenvalues (" << eigenVals.size() << "): " << std::endl
            << eigenVals << std::endl;
  std::cout << " Eigenvecs 1-3 (" << eigenVecs.rows() << " x "
            << eigenVecs.cols() << "): " << std::endl
            << eigenVecs.block(0, 0, eigenVecs.rows(), 3) << std::endl;

  // no dependence on global parameters - these terms only enter the
  // track covariance sub-matrix of the alignment problem (bottom right
  // quadrant)
  globalDeriv.clear();
  globalIndices.clear();

  /// convert each EV to a pseudo-measurement
  for (long iMeas = 0; iMeas < eigenVecs.rows(); ++iMeas) {
    // std::cout <
    if (eigenVals(iMeas) <= 0)
      continue;
    localDeriv.assign(localDeriv.size(), 0.);
    // fill the local derivatives from the current eigenvector
    std::size_t iPar = 0;

    for (std::size_t iTrkPar = 0; iTrkPar < state.trackParametersDim;
         ++iTrkPar) {
      if (kickUsOut.contains(iTrkPar)) {
        continue;
      }
      localDeriv[iPar] = eigenVecs(iPar, iMeas);
      ++iPar;
    }
    // and write a pseudo-measurement to Mille.
    record.addData(
        // residual == 0 for pseudo-measurements
        0,
        // EV == weight = 1/sigma^2
        1. / std::sqrt(eigenVals(iMeas)),
        // local parameter indices
        localIndices,
        // local derivatives
        localDeriv, globalIndices, globalDeriv);
  }
  ++ix;
  // track is fully written - end the record in Mille
  record.writeRecord();
}

Mille::MilleDecoder::ReadResult unpackMilleRecord(
    Mille::IMilleReader& reader,
    ActsAlignment::detail::TrackAlignmentState& targetState,
    const std::unordered_map<const Acts::Surface*, std::size_t>&
        idxedAlignSurfaces) {
  /// book a decoder
  Mille::MilleDecoder decoder;
  // vector to hold the extracted measurements
  std::vector<Mille::MilleMeasurement> measurements;
  // attempt to decode the next record from the binary
  auto res = decoder.decode(reader, measurements);

  // if we are EoF or encountered an error, return the result.
  if (res != Mille::MilleDecoder::ReadResult::OK) {
    return res;
  }

  // initialise the components of the target state which we will touch
  targetState.measurementDim = 0;

  // Still here - we got a valid record! Let's write it into our target state.
  // In the following, we emulate what MillePede-II is doing internally.
  // This is somewhat approximate, as we do not run any of the cleaning /
  // conditioning performed by MP-II.

  // Step 1: Parameter discovery
  // Goal: Identify all existing parameters and assign internal indices.
  int firstLocal = 99999;
  int lastLocal = 0;
  std::set<int> seenGlobalLabels;
  std::set<int> seenSurfaceLabels;
  // a measurement with a residual of identical zero is typically a
  // correlation constraint encoded as pseudo-measurement
  targetState.measurementDim =
      std::count_if(measurements.begin(), measurements.end(),
                    [](const Mille::MilleMeasurement& measurement) {
                      return measurement.measurement != 0;
                    });

  // discover labels in use
  for (const Mille::MilleMeasurement& measurement : measurements) {
    if (measurement.localLabels.empty())
      continue;

    auto [minLabel, maxLabel] = std::minmax_element(
        measurement.localLabels.begin(), measurement.localLabels.end());

    firstLocal = std::min(firstLocal, *minLabel);
    lastLocal = std::max(lastLocal, *maxLabel);

    seenGlobalLabels.insert(measurement.globalLabels.begin(),
                            measurement.globalLabels.end());
  }
  if (lastLocal < firstLocal)
    return Mille::MilleDecoder::ReadResult::error;
  targetState.trackParametersDim = lastLocal - firstLocal + 1;
  targetState.alignmentDof = seenGlobalLabels.size();

  for (int label : seenGlobalLabels) {
    if ((label - 1) % Acts::eAlignmentSize == 0) {
      seenSurfaceLabels.insert((label - 1) / Acts::eAlignmentSize);
    }
  };

  /// the trackAlignmentState uses an internal indexing for alignment
  /// parameters - so remap the indices to replicate this internal logic.
  std::map<int, int> globalToInternal;

  /// try to map indices from the global indexed surface list, if we have it.
  unsigned long iExtra = 0;
  for (auto [surface, index] : idxedAlignSurfaces) {
    if (seenSurfaceLabels.contains(index)) {
      targetState.alignedSurfaces.emplace(surface,
                                          std::make_pair(index, iExtra++));
    }
  }

  // now use this to fill our internal "global to local" mapping function
  for (auto [surface, indices] : targetState.alignedSurfaces) {
    auto [globIx, intIx] = indices;
    for (std::size_t iAli = 0; iAli < Acts::eAlignmentSize; ++iAli) {
      globalToInternal.emplace(globalIndexSurfToParam(globIx, iAli),
                               internalIndexSurfToParam(intIx, iAli));
    }
  }

  // Now we have the needed information to initialise our matrices
  targetState.measurementCovariance = Acts::DynamicMatrix::Zero(
      targetState.measurementDim, targetState.measurementDim);

  targetState.projectionMatrix = Acts::DynamicMatrix::Zero(
      targetState.measurementDim, targetState.trackParametersDim);

  targetState.alignmentToResidualDerivative = Acts::DynamicMatrix::Zero(
      targetState.measurementDim, targetState.alignmentDof);

  targetState.trackParametersCovariance = Acts::DynamicMatrix::Zero(
      targetState.trackParametersDim, targetState.trackParametersDim);

  targetState.residual = Acts::DynamicVector::Zero(targetState.measurementDim);

  /// Second loop - fill the matrices

  std::size_t iMeas = 0;
  for (const auto& measurement : measurements) {
    // need distinction: Measurement on surface vs. correlation term.
    // The reason is that ACTS only counts surface measurements, and stores
    // the correlation information directly in the track parameter covariance.
    // MillePede considers constraints to be additional measurements.
    // A MillePede pseudomeasurement has residual 0 and no global derivatives.
    bool isMeasurementOnSurface = (measurement.measurement != 0 ||
                                   !measurement.globalDerivatives.empty());
    // surface measurements populate the residual vector and measurement
    // covariance matrix
    if (isMeasurementOnSurface) {
      targetState.residual(iMeas) = measurement.measurement;
      targetState.measurementCovariance(iMeas, iMeas) =
          measurement.uncertainty * measurement.uncertainty;
    }
    // loop over all track parameters affecting this measurement
    for (std::size_t iLoc = 0; iLoc < measurement.localLabels.size(); ++iLoc) {
      // find out where to book it in the ACTS matrix
      unsigned int localIndex = measurement.localLabels[iLoc] - firstLocal;
      // if we are a surface measurement, fill the projection matrix
      if (isMeasurementOnSurface)
        targetState.projectionMatrix(iMeas, localIndex) =
            measurement.localDerivatives[iLoc];
      // now fill the covariance matrix by looping over all products of (local)
      // derivatives
      for (std::size_t jLoc = 0; jLoc < measurement.localLabels.size();
           ++jLoc) {
        // again determine where to book the column index
        unsigned int localIndex2 = measurement.localLabels[jLoc] - firstLocal;
        // and update the covariance.
        targetState.trackParametersCovariance(localIndex, localIndex2) +=
            measurement.localDerivatives[iLoc] *
            measurement.localDerivatives[jLoc] / measurement.uncertainty /
            measurement.uncertainty;
      }
    }
    // loop over global (= alignment) parameters affecting the measurement
    for (std::size_t iGlob = 0; iGlob < measurement.globalLabels.size();
         ++iGlob) {
      // find out where to book - here we need to map to the ACTS track-level
      // indexing scheme
      int internalAliIndex = globalToInternal[measurement.globalLabels[iGlob]];
      // and update the alignment-to-residual derivative matrix.
      targetState.alignmentToResidualDerivative(iMeas, internalAliIndex) =
          measurement.globalDerivatives[iGlob];
    }
    // increment the measurement-on-surface index every time we finish
    // processing one.
    if (isMeasurementOnSurface)
      ++iMeas;
  }

  /// (carefully) invert the covariance - upstairs, we filled it as a weight
  /// matrix
  auto solver = targetState.trackParametersCovariance.ldlt();
  targetState.trackParametersCovariance =
      solver.solve(Acts::DynamicMatrix::Identity(
          targetState.trackParametersDim, targetState.trackParametersDim));

  /// and calculate the dependent members
  /// (first and second derivatives, chi2) of the state.
  ActsAlignment::detail::finaliseTrackAlignState(targetState);

  return res;
}

}  // namespace ActsPlugins::ActsToMille
