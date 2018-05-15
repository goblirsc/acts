// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 Acts project team
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

<<<<<<< HEAD:Core/include/Acts/Propagator/AbortList.hpp
#pragma once
=======
#ifndef ACTS_ABORT_LIST_HPP
#define ACTS_ABORT_LIST_HPP
>>>>>>> cc97e037... split of propator and stepper cache:Core/include/ACTS/Propagator/AbortList.hpp

#include "Acts/Propagator/ActionList.hpp"
#include "Acts/Propagator/detail/Extendable.hpp"
#include "Acts/Propagator/detail/abort_condition_signature_check.hpp"
#include "Acts/Propagator/detail/abort_list_implementation.hpp"
#include "Acts/Utilities/detail/MPL/boost_mpl_helper.hpp"
#include "Acts/Utilities/detail/MPL/has_duplicates.hpp"

namespace Acts {

template <typename... conditions>
struct AbortList : private detail::Extendable<conditions...>
{
private:
  static_assert(not detail::has_duplicates_v<conditions...>,
                "same abort conditions type specified several times");

  // clang-format off
  typedef detail::type_collector_t<detail::action_type_extractor, conditions...> actions;
  // clang-format on

  using detail::Extendable<conditions...>::tuple;

public:
  typedef detail::boost_set_as_tparams_t<ActionList, actions> action_list_type;
  using detail::Extendable<conditions...>::get;

  /// This is the call signature for the abort list, it broadcasts the call
  /// to the tuple() memembers of the list
  ///
  /// @tparam propagator_state_t is the state type of the propagator
  /// @tparam stepper_state_t is the state type of the stepper
  /// @tparam result_t is the result type from a certain action
  ///
  /// @param result[in] is the result object from a certin action
  /// @param propState[in,out] is the state object from the propagator
  /// @param stepState[in,out] is the state object from the stepper
  template <typename propagator_state_t,
            typename stepper_state_t,
            typename result_t>
  bool
  operator()(const result_t&     result,
             propagator_state_t& propState,
             stepper_state_t&    stepState) const
  {
    // clang-format off
    static_assert(detail::all_of_v<detail::abort_condition_signature_check_v<
                        conditions, 
                        propagator_state_t, 
                        stepper_state_t>...>,
                  "not all abort conditions support the specified input");
    // clang-format on

    return detail::abort_list_impl<conditions...>::check(
        tuple(), result, propState, stepState);
  }
};

}  // namespace Acts