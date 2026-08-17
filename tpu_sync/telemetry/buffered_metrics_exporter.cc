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

#include "tpu_sync/telemetry/buffered_metrics_exporter.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

namespace {

constexpr absl::string_view kMetricPrefix = "tpu_raiden_";

}  // namespace

BufferedMetricsExporter::BufferedMetricsExporter(
    absl::Span<const MetricMetadata> metrics) {
  for (const auto& meta : metrics) {
    switch (meta.type) {
      case MetricType::kCounter:
        counters_.emplace(meta.name,
                          std::make_unique<LockFreeCounterAccumulator>());
        break;
      case MetricType::kGauge:
        gauges_.emplace(meta.name, std::make_unique<QueueBuffer<>>());
        break;
      case MetricType::kHistogram:
        histograms_.emplace(meta.name, std::make_unique<QueueBuffer<>>());
        break;
    }
  }
}

void BufferedMetricsExporter::IncrementCounter(absl::string_view name,
                                               LabelSpan labels,
                                               uint64_t val) const {
  auto it = counters_.find(name);
  if (it != counters_.end()) {
    it->second->Add(val);
  }
}

void BufferedMetricsExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                       double val) const {
  auto it = gauges_.find(name);
  if (it != gauges_.end()) {
    it->second->Push(val);
  }
}

void BufferedMetricsExporter::ObserveHistogram(absl::string_view name,
                                               LabelSpan labels,
                                               double val) const {
  auto it = histograms_.find(name);
  if (it != histograms_.end()) {
    it->second->Push(val);
  }
}

std::map<std::string, std::vector<double>>
BufferedMetricsExporter::GetAndResetMetricSamples() {
  std::map<std::string, std::vector<double>> result;

  for (const auto& [name, counter] : counters_) {
    uint64_t delta = counter->ExchangeAndReset();
    if (delta > 0) {
      std::string full_name = absl::StrCat(kMetricPrefix, name);
      result[full_name].push_back(static_cast<double>(delta));
    }
  }

  for (const auto& [name, gauge] : gauges_) {
    std::vector<double> samples = gauge->ExtractAndReset();
    if (!samples.empty()) {
      std::string full_name = absl::StrCat(kMetricPrefix, name);
      result[full_name] = std::move(samples);
    }
  }

  for (const auto& [name, histogram] : histograms_) {
    std::vector<double> samples = histogram->ExtractAndReset();
    if (!samples.empty()) {
      std::string full_name = absl::StrCat(kMetricPrefix, name);
      result[full_name] = std::move(samples);
    }
  }

  return result;
}

}  // namespace tpu_raiden::telemetry
