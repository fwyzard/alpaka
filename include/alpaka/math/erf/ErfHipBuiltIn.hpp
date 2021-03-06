/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/math/erf/Traits.hpp>   // Erf



#include <type_traits>
#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 0, 0)
    #include <cuda_runtime_api.h>
#else
    #if BOOST_COMP_HCC
        #include <math_functions.h>
    #else
        #include <math_functions.hpp>
    #endif
#endif

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library erf.
        class ErfHipBuiltIn
        {
        public:
            using ErfBase = ErfHipBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library erf trait specialization.
            template<
                typename TArg>
            struct Erf<
                ErfHipBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                __device__ static auto erf(
                    ErfHipBuiltIn const & /*erf*/,
                    TArg const & arg)
                -> decltype(::erf(arg))
                {
                    //boost::ignore_unused(erf);
                    return ::erf(arg);
                }
            };
        }
    }
}

#endif
