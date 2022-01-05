/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci
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
#    include <alpaka/math/cbrt/CbrtUniformCudaHipBuiltIn.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA cbrt trait specialization.
            template<typename TArg>
            struct Cbrt<CbrtUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_arithmetic<TArg>::value>>
            {
                __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& cbrt_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(cbrt_ctx);
                    return ::cbrt(arg);
                }
            };

            //! The CUDA cbrt float trait specialization.
            template<>
            struct Cbrt<CbrtUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(CbrtUniformCudaHipBuiltIn const& cbrt_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(cbrt_ctx);
                    return ::cbrtf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
