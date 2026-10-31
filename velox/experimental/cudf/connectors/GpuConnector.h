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

namespace facebook::velox::cudf_velox::connector {

/// Implemented by a connector, besides the cuDF Hive and Iceberg ones, whose
/// data sources return CudfVectors, so that its table scans stay on the GPU.
class GpuConnector {
 public:
  virtual ~GpuConnector() = default;
};

} // namespace facebook::velox::cudf_velox::connector
