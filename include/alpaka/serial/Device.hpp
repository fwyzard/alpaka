/**
* Copyright 2014 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of either the GNU General Public License or
* the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License and the GNU Lesser General Public License
* for more details.
*
* You should have received a copy of the GNU General Public License
* and the GNU Lesser General Public License along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/serial/AccSerialFwd.hpp>   // AccSerial

#include <alpaka/host/SystemInfo.hpp>       // host::getCpuName, host::getGlobalMemorySizeBytes

#include <alpaka/interfaces/Device.hpp>     // alpaka::device::DeviceHandle, alpaka::device::DeviceManager

#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits

namespace alpaka
{
    namespace serial
    {
        namespace detail
        {
            // forward declaration
            class DeviceManagerSerial;

            //#############################################################################
            //! The serial accelerator device handle.
            //#############################################################################
            class DeviceHandleSerial
            {
                friend class DeviceManagerSerial;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleSerial() = default;
                //-----------------------------------------------------------------------------
                //! Copy-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleSerial(DeviceHandleSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Move-constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleSerial(DeviceHandleSerial &&) = default;
                //-----------------------------------------------------------------------------
                //! Assignment-operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceHandleSerial & operator=(DeviceHandleSerial const &) = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST ~DeviceHandleSerial() noexcept = default;

                //-----------------------------------------------------------------------------
                //! \return The device properties.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST device::DeviceProperties getProperties() const
                {
                    device::DeviceProperties deviceProperties;

                    deviceProperties.m_sName = host::getCpuName();
                    deviceProperties.m_uiBlockKernelSizeMax = 1;    // TODO: Magic number.
                    deviceProperties.m_v3uiBlockKernelSizePerDimMax = vec<3u>(1u, 1u, 1u);
                    deviceProperties.m_v3uiGridBlockSizePerDimMax = vec<3u>(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max());
                    deviceProperties.m_uiStreamCount = std::numeric_limits<std::size_t>::max();
                    deviceProperties.m_uiExecutionUnitCount = 1;
                    deviceProperties.m_uiGlobalMemorySizeBytes = host::getGlobalMemorySizeBytes();
                    //deviceProperties.m_uiClockFrequencyHz = TODO;

                    return deviceProperties;
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The serial accelerator device handle.
        //#############################################################################
        template<>
        class DeviceHandle<AccSerial> :
            public device::detail::IDeviceHandle<serial::detail::DeviceHandleSerial>
        {
            friend class serial::detail::DeviceManagerSerial;
        private:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle() = default;

        public:
            //-----------------------------------------------------------------------------
            //! Copy-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle(DeviceHandle const &) = default;
            //-----------------------------------------------------------------------------
            //! Move-constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle(DeviceHandle &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment-operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DeviceHandle & operator=(DeviceHandle const &) = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~DeviceHandle() noexcept = default;
        };
    }

    namespace serial
    {
        namespace detail
        {
            //#############################################################################
            //! The serial accelerator device manager.
            //#############################################################################
            class DeviceManagerSerial
            {
            public:
                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST DeviceManagerSerial() = delete;

                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static std::size_t getDeviceCount()
                {
                    return 1;
                }
                //-----------------------------------------------------------------------------
                //! \return The number of devices available.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static device::DeviceHandle<AccSerial> getDeviceHandleByIndex(std::size_t const & uiIndex)
                {
                    std::size_t const uiNumDevices(getDeviceCount());
                    if(uiIndex >= uiNumDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for device " << uiIndex << " because there are only " << uiNumDevices << " serial devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
                //-----------------------------------------------------------------------------
                //! \return The handle to the currently used device.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static device::DeviceHandle<AccSerial> getCurrentDeviceHandle()
                {
                    return {};
                }
                //-----------------------------------------------------------------------------
                //! Sets the device to use with this accelerator.
                //-----------------------------------------------------------------------------
                ALPAKA_FCT_HOST static void setCurrentDevice(device::DeviceHandle<AccSerial> const & )
                {
                    // The code is already running on this device.
                }
            };
        }
    }

    namespace device
    {
        //#############################################################################
        //! The serial accelerator device manager.
        //#############################################################################
        template<>
        class DeviceManager<AccSerial> :
            public detail::IDeviceManager<serial::detail::DeviceManagerSerial>
        {};
    }
}