/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// An OSS front end for NimbleDumpLib, whose full command line tool depends on
// Meta-internal headers.
//
// Usage: nimble_dump_lite <schema|streams|layout|histogram> <file> [stripe]

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/tools/NimbleDumpLib.h"

#include <iostream>
#include <optional>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <schema|streams|layout|histogram> <file> [stripe]\n";
    return 1;
  }
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::velox::filesystems::registerLocalFileSystem();
  const std::string command = argv[1];
  std::optional<uint32_t> stripe;
  if (argc > 3) {
    stripe = static_cast<uint32_t>(std::stoul(argv[3]));
  }
  facebook::nimble::tools::NimbleDumpLib dump{argv[2], false, std::cout};
  if (command == "schema") {
    dump.emitSchema();
  } else if (command == "streams") {
    dump.emitStreams(false, false, true, false, stripe);
  } else if (command == "layout") {
    dump.emitLayout(false, false);
  } else if (command == "histogram") {
    dump.emitHistogram(false, false, stripe);
  } else {
    std::cerr << "Unknown command " << command << "\n";
    return 1;
  }
  return 0;
}
