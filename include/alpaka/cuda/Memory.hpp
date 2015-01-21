/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/traits/Memory.hpp>         // traits::MemCopy
#include <alpaka/traits/Extent.hpp>         // traits::getXXX

#include <alpaka/core/BasicDims.hpp>        // dim::Dim<N>
#include <alpaka/core/RuntimeExtents.hpp>   // extent::RuntimeExtents<TDim>

#include <alpaka/cuda/MemorySpace.hpp>      // MemSpaceCuda
#include <alpaka/host/MemorySpace.hpp>      // MemSpaceHost

#include <alpaka/cuda/Common.hpp>

#include <cstddef>                          // std::size_t
#include <cassert>                          // assert
#include <memory>                           // std::shared_ptr

namespace alpaka
{
    namespace cuda
    {
        namespace detail
        {
            //#############################################################################
            //! The CUDA memory buffer.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            class MemBufCuda :
                public alpaka::extent::RuntimeExtents<TDim>
            {
            public:
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                template<
                    typename TExtent>
                MemBufCuda(
                    TElem * pMem,
                    std::size_t const & uiPitchBytes,
                    TExtent const & extent) :
                        RuntimeExtents<TDim>(extent),
                        m_spMem(
                            pMem,
                            [](TElem * pBuffer)
                            {
                                assert(pBuffer);
                                cudaFree(reinterpret_cast<void *>(pBuffer));
                            }),
                        m_uiPitchBytes(uiPitchBytes)
                {
                    static_assert(
                        std::is_same<TDim, alpaka::dim::GetDimT<TExtent>>::value,
                        "The extent is required to have the same dimensionality as the MemBufCuda!");
                }

            public:
                std::shared_ptr<TElem> m_spMem;

                std::size_t m_uiPitchBytes; // \TODO: By using class specialization for Dim1 we could remove this value in this case.
            };

            //#############################################################################
            //! Page-locks the memory range specified.
            //#############################################################################
            template<
                typename TMemBuf>
            void pageLockHostMem(
                TMemBuf const & memBuf)
            {
                // cudaHostRegisterDefault: 
                //  See http://cgi.cs.indiana.edu/~nhusted/dokuwiki/doku.php?id=programming:cudaperformance1
                // cudaHostRegisterPortable: 
                //  The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.
                // cudaHostRegisterMapped: 
                //  Maps the allocation into the CUDA address space.The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer().
                //  This feature is available only on GPUs with compute capability greater than or equal to 1.1.
                ALPAKA_CUDA_CHECK(
                    cudaHostRegister(
                        const_cast<void *>(reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf))), 
                        alpaka::extent::getProductOfExtents(memBuf) * sizeof(alpaka::memory::GetMemElemT<TMemBuf>),
                        cudaHostRegisterDefault));
            }
            //#############################################################################
            //! Unmaps page-locked memory.
            //#############################################################################
            template<
                typename TMemBuf>
            void unPageLockHostMem(
                TMemBuf const & memBuf)
            {
                ALPAKA_CUDA_CHECK(
                    cudaHostUnregister(
                        const_cast<void *>(reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf)))));
            }

            //#############################################################################
            //! The CUDA memory copy trait.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopyCuda;
            //#############################################################################
            //! The CUDA 1D memory copy trait specialization.
            //#############################################################################
            template<>
            struct MemCopyCuda<
                alpaka::dim::Dim1>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::memory::GetMemElemT<TMemBufDst>, alpaka::memory::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentWidth <= uiSrcWidth);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBufDst)),
                            reinterpret_cast<void const *>(alpaka::memory::getNativePtr(memBufSrc)),
                            uiExtentWidth * sizeof(alpaka::memory::GetMemElemT<TMemBufDst>),
                            cudaMemcpyKindVal));
                }
            };
            //#############################################################################
            //! The CUDA 2D memory copy trait specialization.
            //#############################################################################
            template<>
            struct MemCopyCuda<
                alpaka::dim::Dim2>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::memory::GetMemElemT<TMemBufDst>, alpaka::memory::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy2D(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBufDst)),
                            alpaka::memory::getPitchBytes(memBufDst),
                            reinterpret_cast<void const *>(alpaka::memory::getNativePtr(memBufSrc)),
                            alpaka::memory::getPitchBytes(memBufSrc),
                            uiExtentWidth * sizeof(alpaka::memory::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            cudaMemcpyKindVal));
                }
            };
            //#############################################################################
            //! The CUDA 3D memory copy trait specialization.
            //#############################################################################
            template<>
            struct MemCopyCuda<
                alpaka::dim::Dim3>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopyCuda(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent, 
                    cudaMemcpyKind const & cudaMemcpyKindVal)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBufDst>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");
                    static_assert(
                        std::is_same<alpaka::memory::GetMemElemT<TMemBufDst>, alpaka::memory::GetMemElemT<TMemBufSrc>>::value,
                        "The source and the destination buffers are required to have the same element type!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBufDst));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBufDst));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBufDst));
                    auto const uiSrcWidth(alpaka::extent::getWidth(memBufSrc));
                    auto const uiSrcHeight(alpaka::extent::getHeight(memBufSrc));
                    auto const uiSrcDepth(alpaka::extent::getDepth(memBufSrc));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);
                    assert(uiExtentWidth <= uiSrcWidth);
                    assert(uiExtentHeight <= uiSrcHeight);
                    assert(uiExtentDepth <= uiSrcDepth);

                    // Fill CUDA parameter structure.
                    cudaMemcpy3DParms cudaMemcpy3DParmsVal = {0};
                    //cudaMemcpy3DParmsVal.srcArray;    // Either srcArray or srcPtr.
                    //cudaMemcpy3DParmsVal.srcPos;      // Optional. Offset in bytes.
                    cudaMemcpy3DParmsVal.srcPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBufSrc)),
                            alpaka::memory::getPitchBytes(memBufSrc),
                            uiSrcWidth,
                            uiSrcHeight);
                    //cudaMemcpy3DParmsVal.dstArray;    // Either dstArray or dstPtr.
                    //cudaMemcpy3DParmsVal.dstPos;      // Optional. Offset in bytes.
                    cudaMemcpy3DParmsVal.dstPtr = 
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBufDst)),
                            alpaka::memory::getPitchBytes(memBufDst),
                            uiDstWidth,
                            uiDstHeight);
                    cudaMemcpy3DParmsVal.extent = 
                        make_cudaExtent(
                            uiExtentWidth * sizeof(alpaka::memory::GetMemElemT<TMemBufDst>),
                            uiExtentHeight,
                            uiExtentDepth);
                    cudaMemcpy3DParmsVal.kind = cudaMemcpyKindVal;

                    // Initiate the memory copy.
                    ALPAKA_CUDA_CHECK(
                        cudaMemcpy3D(
                            &cudaMemcpy3DParmsVall));
                }
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for host::detail::MemBufCuda.
    //-----------------------------------------------------------------------------
    namespace traits
    {
        namespace dim
        {
            //#############################################################################
            //! The MemBufHost dimension getter trait.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetDim<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = TDim;
            };
        }

        namespace extent
        {
            //#############################################################################
            //! The MemBufCuda width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetWidth<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 1u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getWidth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiWidth;
                }
            };

            //#############################################################################
            //! The MemBufCuda height get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetHeight<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 2u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getHeight(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiHeight;
                }
            };
            //#############################################################################
            //! The MemBufCuda depth get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetDepth<
                cuda::detail::MemBufCuda<TElem, TDim>,
                typename std::enable_if<(TDim::value >= 3u) && (TDim::value <= 3u)>::type>
            {
                static std::size_t getDepth(
                    cuda::detail::MemBufCuda<TElem, TDim> const & extent)
                {
                    return extent.m_uiDepth;
                }
            };
        }

        namespace memory
        {
            //#############################################################################
            //! The MemBufCuda memory space trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemSpace<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = alpaka::memory::MemSpaceCuda;
            };

            //#############################################################################
            //! The MemBufCuda memory element type get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetMemElem<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                using type = TElem;
            };

            //#############################################################################
            //! The MemBufCuda memory buffer type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetMemBuf<
                TElem, TDim, alpaka::memory::MemSpaceCuda>
            {
                using type = host::detail::MemBufCuda<TElem, TDim>;
            };

            //#############################################################################
            //! The MemBufCuda native pointer get trait specialization.
            //#############################################################################
            template<
                typename TElem, 
                typename TDim>
            struct GetNativePtr<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                static TElem const * getNativePtr(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
                static TElem * getNativePtr(
                    cuda::detail::MemBufCuda<TElem, TDim> & memBuf)
                {
                    return memBuf.m_spMem.get();
                }
            };

            //#############################################################################
            //! The CUDA buffer pitch get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TDim>
            struct GetPitchBytes<
                cuda::detail::MemBufCuda<TElem, TDim>>
            {
                static std::size_t getPitchBytes(
                    cuda::detail::MemBufCuda<TElem, TDim> const & memPitch)
                {
                    return memPitch.m_uiPitchBytes;
                }
            };

            //#############################################################################
            //! The CUDA 1D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim1, 
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1> memAlloc(
                    TExtent const & extent)
                {
                    auto const uiWidth(extent::getWidth(extent));
                    assert(uiWidth>0);
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);

                    void * pBuffer;
                    ALPAKA_CUDA_CHECK(cudaMalloc(
                        &pBuffer, 
                        uiWidthBytes));
                    assert((pBuffer));

                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim1>(
                            pBuffer,
                            uiWidthBytes,
                            extent);
                }
            };

            //#############################################################################
            //! The CUDA 2D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim2, 
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2> memAlloc(
                    TExtent const & extent)
                {
                    auto const uiWidth(extent::getWidth(extent));
                    auto const uiWidthBytes(uiWidth * sizeof(T));
                    assert(uiWidthBytes>0);
                    auto const uiHeight(extent::getHeight(extent));
                    auto const uiElementCount(uiWidth * uiHeight;
                    assert(uiElementCount>0);

                    void * pBuffer;
                    int iPitch;
                    ALPAKA_CUDA_CHECK(cudaMallocPitch(
                        &pBuffer,
                        &iPitch,
                        uiWidthBytes,
                        uiHeight));
                    assert(pBuffer);
                    assert(iPitch>uiWidthBytes);
                    
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim2>(
                            pBuffer,
                            static_cast<std::size_t>(iPitch),
                            extent);
                }
            };

            //#############################################################################
            //! The CUDA 3D memory allocation trait specialization.
            //#############################################################################
            template<typename T>
            struct MemAlloc<
                T, 
                alpaka::dim::Dim3, 
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent>
                static alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3> memAlloc(
                    TExtent const & extent)
                {
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            extent::getWidth(extent),
                            extent::getHeight(extent),
                            extent::getDepth(extent)));
                    
                    cudaPitchedPtr cudaPitchedPtrVal;
                    ALPAKA_CUDA_CHECK(cudaMalloc3D(
                        &cudaPitchedPtrVal,
                        cudaExtentVal));

                    assert(cudaPitchedPtrVal.ptr);
                    
                    return
                        alpaka::cuda::detail::MemBufCuda<T, alpaka::dim::Dim3>(
                            cudaPitchedPtrVal.ptr,
                            static_cast<std::size_t>(cudaPitchedPtrVal.pitch),
                            extent);
                }
            };

            //#############################################################################
            //! The CUDA to Host memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim,
                alpaka::memory::MemSpaceHost,
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst,
                    TMemBufSrc const & memBufSrc,
                    TExtent const & extent)
                {
                    cuda::detail::pageLockHostMem(memBufDst);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extent,
                        cudaMemcpyDeviceToHost);

                    cuda::detail::unPageLockHostMem(memBufDst);
                }
            };
            //#############################################################################
            //! The Host to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim,
                alpaka::memory::MemSpaceCuda,
                alpaka::memory::MemSpaceHost>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent)
                {
                    cuda::detail::pageLockHostMem(memBufSrc);

                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extent,
                        cudaMemcpyHostToDevice);

                    cuda::detail::unPageLockHostMem(memBufSrc);
                }
            };
            //#############################################################################
            //! The CUDA to CUDA memory copy trait specialization.
            //#############################################################################
            template<
                typename TDim>
            struct MemCopy<
                TDim,
                alpaka::memory::MemSpaceCuda,
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent, 
                    typename TMemBufSrc, 
                    typename TMemBufDst>
                static void memCopy(
                    TMemBufDst & memBufDst, 
                    TMemBufSrc const & memBufSrc, 
                    TExtent const & extent)
                {
                    alpaka::cuda::detail::MemCopyCuda<TDim>::memCopyCuda(
                        memBufDst,
                        memBufSrc,
                        extent,
                        cudaMemcpyDeviceToDevice);
                }
            };

            //#############################################################################
            //! The CUDA 1D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim1,
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtent const & extent)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim1>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim1 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf)),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::memory::GetMemElemT<TMemBuf>)));
                }
            };
            //#############################################################################
            //! The CUDA 2D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim2,
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtent const & extent)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim2>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim2 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    alpaka::extent::RuntimeExtent
                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);

                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset2D(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf)),
                            alpaka::memory::getPitchBytes(memBuf),
                            static_cast<int>(byte),
                            uiExtentWidth * sizeof(alpaka::memory::GetMemElemT<TMemBuf>),
                            uiExtentHeight));
                }
            };
            //#############################################################################
            //! The CUDA 3D memory set trait specialization.
            //#############################################################################
            template<>
            struct MemSet<
                alpaka::dim::Dim3,
                alpaka::memory::MemSpaceCuda>
            {
                template<
                    typename TExtent, 
                    typename TMemBuf>
                static void memSet(
                    TMemBuf & memBuf, 
                    std::uint8_t const & byte, 
                    TExtent const & extent)
                {
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::Dim3>::value,
                        "The destination buffer is required to have the dimensionality alpaka::dim::Dim3 for this specialization!");
                    static_assert(
                        std::is_same<alpaka::dim::GetDimT<TMemBuf>, alpaka::dim::GetDimT<TExtent>>::value,
                        "The destination buffer and the extent are required to have the same dimensionality!");

                    auto const uiExtentWidth(alpaka::extent::getWidth(extent));
                    auto const uiExtentHeight(alpaka::extent::getHeight(extent));
                    auto const uiExtentDepth(alpaka::extent::getDepth(extent));
                    auto const uiDstWidth(alpaka::extent::getWidth(memBuf));
                    auto const uiDstHeight(alpaka::extent::getHeight(memBuf));
                    auto const uiDstDepth(alpaka::extent::getDepth(memBuf));
                    assert(uiExtentWidth <= uiDstWidth);
                    assert(uiExtentHeight <= uiDstHeight);
                    assert(uiExtentDepth <= uiDstDepth);

                    // Fill CUDA parameter structures.
                    cudaPitchedPtr const cudaPitchedPtrVal(
                        make_cudaPitchedPtr(
                            reinterpret_cast<void *>(alpaka::memory::getNativePtr(memBuf)),
                            alpaka::memory::getPitchBytes(memBuf),
                            uiDstWidth,
                            uiDstHeight));
                    
                    cudaExtent const cudaExtentVal(
                        make_cudaExtent(
                            uiExtentWidth,
                            uiExtentHeight,
                            uiExtentDepth));
                    
                    // Initiate the memory set.
                    ALPAKA_CUDA_CHECK(
                        cudaMemset3D(
                            cudaPitchedPtrVal,
                            static_cast<int>(byte),
                            cudaExtentVal));
                }
            };
        }
    }
}
