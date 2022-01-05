/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Andrea Bocci
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
#    include <alpaka/math/exp/ExpUniformCudaHipBuiltIn.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace math
    {
        namespace traits
        {
            //! The CUDA exp trait specialization.
            template<typename TArg>
            struct Exp<ExpUniformCudaHipBuiltIn, TArg, std::enable_if_t<std::is_floating_point<TArg>::value>>
            {
                __device__ auto operator()(ExpUniformCudaHipBuiltIn const& exp_ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(exp_ctx);
                    return ::exp(arg);
                }
            };

            //! The CUDA exp float specialization.
            template<>
            struct Exp<ExpUniformCudaHipBuiltIn, float>
            {
                __device__ auto operator()(ExpUniformCudaHipBuiltIn const& exp_ctx, float const& arg) -> float
                {
                    alpaka::ignore_unused(exp_ctx);
                    return ::expf(arg);
                }
            };
        } // namespace traits
    } // namespace math
} // namespace alpaka

#endif
