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

#include <string>

// Versions of the fused queries that Meld compiles and runs on the GPU with
// its own code generator, without cuDF. Meld is loaded from its shared
// library rather than linked, since it brings an Arrow and an LLVM of its
// own. Its plans read the HBM cache in place: the cached copy of every
// column is adopted for the pinned host buffer it was made from.

namespace facebook::velox::supervelox::meld_backend {

/// Loads libmeld-velox.so from `library` and registers the Meld versions of
/// the queries. Returns false, with the reason in `error`, if it cannot.
bool registerMeldBackend(const std::string& library, std::string& error);

} // namespace facebook::velox::supervelox::meld_backend
