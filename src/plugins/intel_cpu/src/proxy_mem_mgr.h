// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

class ProxyMemoryMngr : public IMemoryMngrObserver {
public:
    ProxyMemoryMngr() : m_pOrigMngr(std::make_shared<DnnlMemoryMngr>(make_unique<MemoryMngrWithReuse>()), m_pMngr(m_pOrigMngr) {}
    explicit ProxyMemoryMngr(MemoryMngrPtr pMngr) : ProxyMemoryMngr() {
        OPENVINO_ASSERT(pMngr, "Memory manager is uninitialized");
        m_pMngr = pMngr;
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

    void setManager(MemoryMngrPtr _pMngr);

private:
    // We keep the original MemMngr as may fallback to copy output.
    const MemoryMngrPtr m_pOrigMngr;
    MemoryMngrPtr m_pMngr;

    // WA: resize stage might not work because there is no shape change,
    // but the underlying actual memory manager changes.
    size_t m_size = 0ul;
};
using ProxyMemoryMngrPtr = std::shared_ptr<ProxyMemoryMngr>;
using ProxyMemoryMngrCPtr = std::shared_ptr<const ProxyMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov