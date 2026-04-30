/* Copyright (c) V-Nova International Limited 2024. All rights reserved.
 * This software is licensed under the BSD-3-Clause-Clear License.
 * No patent licenses are granted under this license. For enquiries about patent licenses,
 * please contact legal@v-nova.com.
 * The LCEVCdec software is a stand-alone project and is NOT A CONTRIBUTION to any other project.
 * If the software is incorporated into another project, THE TERMS OF THE BSD-3-CLAUSE-CLEAR LICENSE
 * AND THE ADDITIONAL LICENSING INFORMATION CONTAINED IN THIS FILE MUST BE MAINTAINED, AND THE
 * SOFTWARE DOES NOT AND MUST NOT ADOPT THE LICENSE OF THE INCORPORATING PROJECT. However, the
 * software may be incorporated into a project under a compatible license provided the requirements
 * of the BSD-3-Clause-Clear license are respected, and V-Nova International Limited remains
 * licensor of the software ONLY UNDER the BSD-3-Clause-Clear license (not the compatible license).
 * ANY ONWARD DISTRIBUTION, WHETHER STAND-ALONE OR AS PART OF ANY OTHER PROJECT, REMAINS SUBJECT TO
 * THE EXCLUSION OF PATENT LICENSES PROVISION OF THE BSD-3-CLAUSE-CLEAR LICENSE. */

#ifndef VN_LCEVC_COMMON_TYPE_ARENA_HPP
#define VN_LCEVC_COMMON_TYPE_ARENA_HPP

#include <LCEVC/common/class_utils.hpp>
#include <LCEVC/common/vector.h>
#include <LCEVC/common/vector.hpp>
//
#include <utility>

namespace lcevc_dec::common {

// Templated pool allocator - based on 'Vector'
//
// NB: This does not grow the underlying vector (For address stability)
//
template <typename T>
class FreePool
{
public:
    explicit FreePool(uint32_t reserved, LdcMemoryAllocator* allocator)
        : m_pool(reserved, allocator)
        , m_free(reserved, allocator)
    {
        // Build free list of all reserved slots
        for (uint32_t i = 0; i < reserved; ++i) {
            m_free.append(m_pool.grow());
        }
    }

    explicit FreePool(uint32_t reserved)
        : m_pool(reserved)
        , m_free(reserved)
    {
        // Build free list of all reserved slots
        for (uint32_t i = 0; i < reserved; ++i) {
            m_free.append(m_pool.grow());
        }
    }

    ~FreePool() {}

    uint32_t capacity() const { return m_pool.size(); }
    uint32_t size() const { return m_free.size(); }

    /*! Obtain an uninitialized slot for a T and return its pointer.
     *
     * return  nullptr if no free slots
     */
    T* allocate()
    {
        if (!m_free.isEmpty()) {
            T* p = m_free.back();
            m_free.shrink();
            return p;
        }

        return nullptr;
    }

    /*! Return an object to the free list.
     */
    void release(T* p) noexcept
    {
        if (!p) {
            return;
        }

        // Put the storage back on the free list.
        m_free.append(p);
    }

    /*! Get a slot, and construct it with given arguments
     *
     * return pointer to newly constructed slot
     */
    template <typename... Args>
    T* make(Args&&... args)
    {
        T* p = allocate();
        if(!p) {
            return nullptr;
        }
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
        return p;
    }

    /*! Destruct object in slot and return slot to free list
     */
    void destroy(T* p) noexcept
    {
        if (!p) {
            return;
        }

        // Call destructor
        p->~T();

        release(p);
    }

    VNNoCopyNoMove(FreePool);

private:
    // The objects
    Vector<T> m_pool{};

    // List of free objects
    Vector<T*> m_free{};
};

} // namespace lcevc_dec::common

#endif // VN_LCEVC_COMMON_VECTOR_H
