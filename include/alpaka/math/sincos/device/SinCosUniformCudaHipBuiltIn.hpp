/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/CudaHipMath.hpp>
#    include <alpaka/core/Unused.hpp>
#    include <alpaka/math/sincos/SinCosUniformCudaHipBuiltIn.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA sincos trait specialization.
            template<>
            struct SinCos<SinCosUniformCudaHipBuiltIn, double>
            {
                __device__ auto operator()(
                    SinCosUniformCudaHipBuiltIn const& sincos_ctx,
                    double const& arg,
                    double& result_sin,
                    double& result_cos) -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincos(arg, &result_sin, &result_cos);
                }
            };

            //! The CUDA sincos float specialization.
            template<>
            struct SinCos<SinCosUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(
                    SinCosUniformCudaHipBuiltIn const& sincos_ctx,
                    float const& arg,
                    float& result_sin,
                    float& result_cos) -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    ::sincosf(arg, &result_sin, &result_cos);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
