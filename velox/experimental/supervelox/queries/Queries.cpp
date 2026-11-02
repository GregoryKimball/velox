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
#include "velox/experimental/supervelox/queries/Query.h"

#include <stdexcept>

namespace facebook::velox::supervelox {

QuerySpec q1Spec();
QuerySpec q2Spec();
QuerySpec q3Spec();
QuerySpec q4Spec();
QuerySpec q5Spec();
QuerySpec q6Spec();
QuerySpec q7Spec();
QuerySpec q8Spec();
QuerySpec q9Spec();
QuerySpec q10Spec();
QuerySpec q11Spec();
QuerySpec q12Spec();
QuerySpec q13Spec();
QuerySpec q14Spec();
QuerySpec q15Spec();
QuerySpec q16Spec();
QuerySpec q17Spec();
QuerySpec q18Spec();
QuerySpec q19Spec();
QuerySpec q20Spec();
QuerySpec q21Spec();
QuerySpec q22Spec();

const std::vector<QuerySpec>& querySpecs() {
  static const std::vector<QuerySpec> specs = {
      q1Spec(),  q2Spec(),  q3Spec(),  q4Spec(),  q5Spec(),  q6Spec(),  q7Spec(),  q8Spec(),
      q9Spec(),  q10Spec(), q11Spec(), q12Spec(), q13Spec(), q14Spec(), q15Spec(), q16Spec(),
      q17Spec(), q18Spec(), q19Spec(), q20Spec(), q21Spec(), q22Spec(),
  };
  return specs;
}

const QuerySpec* findQuery(int32_t number) {
  for (const auto& spec : querySpecs()) {
    if (spec.number == number) {
      return &spec;
    }
  }
  return nullptr;
}

namespace {

std::map<std::pair<Backend, int32_t>, QueryFunction>& implementations() {
  static std::map<std::pair<Backend, int32_t>, QueryFunction> functions;
  return functions;
}

} // namespace

const char* backendName(Backend backend) {
  switch (backend) {
    case Backend::kHandwritten:
      return "handwritten";
    case Backend::kCudf:
      return "cudf";
    case Backend::kMeld:
      return "meld";
  }
  return "?";
}

Backend parseBackend(const std::string& name) {
  for (Backend backend : {Backend::kHandwritten, Backend::kCudf, Backend::kMeld}) {
    if (name == backendName(backend)) {
      return backend;
    }
  }
  throw std::invalid_argument("unknown backend " + name);
}

void registerImplementation(Backend backend, int32_t query, QueryFunction function) {
  implementations()[{backend, query}] = function;
}

QueryFunction implementation(const QuerySpec& spec, Backend backend) {
  if (backend == Backend::kHandwritten) {
    return spec.run;
  }
  auto it = implementations().find({backend, spec.number});
  return it == implementations().end() ? nullptr : it->second;
}

} // namespace facebook::velox::supervelox
