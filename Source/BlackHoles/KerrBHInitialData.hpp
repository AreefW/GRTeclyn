/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef KERRBHINITIALDATA_HPP_
#define KERRBHINITIALDATA_HPP_


#include "CoordinateTransformations.hpp"
#include "Coordinates.hpp"
#include "StateVariables.hpp" //This files needs NUM_VARS - total number of components
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include <AMReX_GpuQualifiers.H>
#include <array>

//! Class which computes the Kerr initial conditions per arXiv 1401.1548
class KerrBHInitialData
{
  public:
    //! Struct for the params of the Kerr BHInitialData
    struct params_t
    {
        amrex::Real mass; //!<< The mass of the Kerr BH
        std::array<amrex::Real, AMREX_SPACEDIM> center; //!< The center of the Kerr BH
        amrex::Real spin; //!< The spin param a = J/M, so 0 <= |a| <= M
    };

  protected:
    amrex::Real m_dx;
    params_t m_params;

  public:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_FORCE_INLINE
    KerrBHInitialData(params_t a_params, amrex::Real a_dx)
        : m_dx(a_dx), m_params(a_params)
    {
        // check this spin param is sensible
        if (std::abs(m_params.spin) > m_params.mass)
        {
            amrex::Abort("The spin parameter must satisfy |a| <= M");
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    AMREX_GPU_DEVICE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state) const;

  protected:
    //! Function which computes the components of the metric in spherical coords
    AMREX_FORCE_INLINE AMREX_GPU_DEVICE void 
    compute_kerr(Tensor::Rank2 &spherical_g, //!<< The spatial metric in spherical coords
                 Tensor::Rank2 &spherical_K, //!<< The extrinsic curvature in spherical coords
                 Tensor::Rank1 &spherical_shift, //!<< The spherical components of the shift
                 amrex::Real &kerr_lapse, //!<< The lapse for the kerr solution
                 const Coordinates coords) const;
};

#include "KerrBHInitialData.impl.hpp"

#endif /* KERRBHINITIALDATA_HPP_ */
