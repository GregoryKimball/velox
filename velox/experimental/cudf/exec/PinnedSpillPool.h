/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "velox/experimental/cudf/exec/HostSpillBudget.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace facebook::velox::cudf_velox {

// Owned CCCL host pool with a requested CUDA maximum, including cached blocks.
// CUDA may round that physical maximum upward for alignment. HostSpillBudget
// enforces the exact logical limit; deployment headroom must cover rounding.
// reservedBytes() reports actual physical retention. No default pool is modified.
class PinnedSpillPool : public std::enable_shared_from_this<PinnedSpillPool> {
 public:
  class Allocation;
  // Independent worker/query admission happens before acquiring codec workspace.
  // Spill payloads never enter the ordinary CPU memory-pool hierarchy.
  class Reservation {
   public:
    Reservation() = default;
    ~Reservation();
    Reservation(Reservation&&) noexcept;
    Reservation& operator=(Reservation&&) noexcept;
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Allocation allocate(std::size_t bytes);
    uint64_t bytes() const {
      return admitted_.bytes();
    }

   private:
    friend class PinnedSpillPool;
    Reservation(
        std::shared_ptr<PinnedSpillPool> owner,
        HostSpillBudget::Reservation admitted) noexcept;
    void release() noexcept;
    std::shared_ptr<PinnedSpillPool> owner_;
    HostSpillBudget::Reservation admitted_;
  };

  class Allocation {
   public:
    Allocation() = default;
    ~Allocation();
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    void* data() const {
      return data_;
    }
    std::size_t size() const {
      return bytes_;
    }

   private:
    friend class PinnedSpillPool;
    Allocation(
        std::shared_ptr<PinnedSpillPool> owner,
        HostSpillBudget::Reservation charge,
        void* data,
        std::size_t bytes) noexcept;
    void reset() noexcept;
    std::shared_ptr<PinnedSpillPool> owner_;
    HostSpillBudget::Reservation charge_;
    void* data_{nullptr};
    std::size_t bytes_{0};
  };

  static std::shared_ptr<PinnedSpillPool> create(
      std::size_t limitBytes,
      std::optional<int> numaNode = std::nullopt);
  ~PinnedSpillPool();
  const std::shared_ptr<HostSpillBudget>& budget() const {
    return budget_;
  }
  std::size_t reservedBytes() const;

  // Requires admission before allocation. CUDA users must finish accessing an
  // Allocation before destroying it, just like the underlying CCCL resource.
  Reservation reserve(
      const std::shared_ptr<HostSpillBudget>& queryBudget,
      std::size_t bytes);

 private:
  struct Impl;
  Allocation allocate(Reservation& admitted, std::size_t bytes);
  PinnedSpillPool(std::size_t limitBytes, std::optional<int> numaNode);
  std::shared_ptr<HostSpillBudget> budget_;
  std::unique_ptr<Impl> impl_;
};
} // namespace facebook::velox::cudf_velox
