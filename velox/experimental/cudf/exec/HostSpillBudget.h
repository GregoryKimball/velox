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

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace facebook::velox::cudf_velox {

// Admission ledger for live spill allocations and in-flight reservations.
// Unlike MemoryPool reservation growth, its hard limit is enforced even while
// GPU arbitration is running. It never initiates arbitration or spills to disk.
class HostSpillBudget : public std::enable_shared_from_this<HostSpillBudget> {
 public:
  class Reservation {
   public:
    Reservation() = default;
    ~Reservation();
    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    // Transfer admitted bytes to a retained allocation without changing the
    // aggregate charge. Remaining bytes are released on destruction.
    Reservation split(uint64_t bytes);
    uint64_t bytes() const {
      return bytes_;
    }
    bool belongsTo(const HostSpillBudget* worker) const;

   private:
    friend class HostSpillBudget;
    Reservation(
        std::array<std::shared_ptr<HostSpillBudget>, 2> budgets,
        uint64_t bytes) noexcept;
    void release() noexcept;
    std::array<std::shared_ptr<HostSpillBudget>, 2> budgets_;
    uint64_t bytes_{0};
  };

  static std::shared_ptr<HostSpillBudget> createWorker(uint64_t limitBytes);
  std::shared_ptr<HostSpillBudget> createQuery(
      std::string queryId,
      uint64_t limitBytes);
  Reservation reserve(uint64_t bytes);
  uint64_t limit() const {
    return limit_;
  }
  uint64_t used() const;
  uint64_t peak() const;

 private:
  HostSpillBudget(
      std::string name,
      uint64_t limit,
      std::shared_ptr<HostSpillBudget> parent = nullptr);
  void checkLocked(uint64_t bytes) const;
  const std::string name_;
  const uint64_t limit_;
  const std::shared_ptr<HostSpillBudget> parent_;
  mutable std::mutex mutex_;
  uint64_t used_{0};
  uint64_t peak_{0};
};

} // namespace facebook::velox::cudf_velox
