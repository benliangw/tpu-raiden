// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_MOCK_METRICS_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_MOCK_METRICS_BACKEND_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_api.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

// Mock MetricsBackend for testing.
class MockMetricsBackend : public MetricsBackend {
 public:
  MOCK_METHOD(void, IncrementCounter,
              (absl::string_view name, LabelSpan labels, uint64_t val),
              (override, const));
  MOCK_METHOD(void, SetGauge,
              (absl::string_view name, LabelSpan labels, double val),
              (override, const));
  MOCK_METHOD(void, ObserveHistogram,
              (absl::string_view name, LabelSpan labels, double val),
              (override, const));
  MOCK_METHOD(std::string, GetTextSnapshot, (), (override, const));
  MOCK_METHOD((std::map<std::string, std::vector<double>>),
              GetAndResetMetricSamples, (), (override));
};

// Helper class to reset the global metrics backend to empty in a scoped
// manner.
class ScopedMetricsBackendReset {
 public:
  ScopedMetricsBackendReset() = default;

  explicit ScopedMetricsBackendReset(std::unique_ptr<MetricsBackend> backend) {
    std::vector<std::unique_ptr<MetricsBackend>> backends;
    if (backend != nullptr) {
      backends.push_back(std::move(backend));
    }
    RaidenMetricStore::GetGlobalMetricStore().SetBackends(std::move(backends));
  }

  explicit ScopedMetricsBackendReset(
      std::vector<std::unique_ptr<MetricsBackend>> backends) {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends(std::move(backends));
  }

  ~ScopedMetricsBackendReset() {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }

  ScopedMetricsBackendReset(const ScopedMetricsBackendReset&) = delete;
  ScopedMetricsBackendReset& operator=(const ScopedMetricsBackendReset&) =
      delete;
  ScopedMetricsBackendReset(ScopedMetricsBackendReset&&) = delete;
  ScopedMetricsBackendReset& operator=(ScopedMetricsBackendReset&&) = delete;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_MOCK_METRICS_BACKEND_H_
