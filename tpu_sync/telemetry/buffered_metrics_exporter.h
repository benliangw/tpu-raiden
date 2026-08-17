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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {

inline constexpr size_t kDefaultQueueBufferSize = 4096;

// Fixed-capacity sample buffer for step-level metric observations.
// Uses an absl::Mutex to ensure thread safety with O(1) buffer swapping
// on extraction.
template <size_t N = kDefaultQueueBufferSize>
class QueueBuffer {
 public:
  explicit QueueBuffer(size_t max_capacity = N)
      : max_capacity_(max_capacity) {}

  // Adds a value to the buffer. If the buffer is full, the value is dropped.
  void Push(double val) {
    absl::MutexLock lock(mutex_);
    if (samples_.size() < max_capacity_) {
      samples_.push_back(val);
    }
    // else: drop the value. Figure out if we want to log this.
  }

  // Returns the values in the buffer and resets the buffer to empty.
  // The returned vector will contain at most max_capacity_ values.
  std::vector<double> ExtractAndReset() {
    std::vector<double> extracted;
    {
      absl::MutexLock lock(mutex_);
      extracted.swap(samples_);
    }
    return extracted;
  }

 private:
  const size_t max_capacity_;
  mutable absl::Mutex mutex_;
  std::vector<double> samples_ ABSL_GUARDED_BY(mutex_);
};

// Accumulator for counter deltas.
class LockFreeCounterAccumulator {
 public:
  // Adds a value to the counter.
  void Add(uint64_t val) { value_.fetch_add(val, std::memory_order_relaxed); }

  // Returns the current value of the counter and resets it to 0.
  uint64_t ExchangeAndReset() {
    return value_.exchange(0, std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t> value_{0};
};

// MetricsBackend for step-level sample buffering.
//
// Thread Safety:
// IncrementCounter(), SetGauge(), and ObserveHistogram() are thread-safe.
// Counters use atomic fetch-add, while gauges and histograms use fine-grained
// mutex-protected sample buffers.
class BufferedMetricsExporter : public MetricsBackend {
 public:
  explicit BufferedMetricsExporter(
      absl::Span<const MetricMetadata> metrics = metric_metadata::kAllMetrics);
  ~BufferedMetricsExporter() override = default;

  BufferedMetricsExporter(const BufferedMetricsExporter&) = delete;
  BufferedMetricsExporter& operator=(const BufferedMetricsExporter&) = delete;
  BufferedMetricsExporter(BufferedMetricsExporter&&) = delete;
  BufferedMetricsExporter& operator=(BufferedMetricsExporter&&) = delete;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override;

  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;

  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  std::string GetTextSnapshot() const override { return ""; }

  std::map<std::string, std::vector<double>> GetAndResetMetricSamples()
      override;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<LockFreeCounterAccumulator>>
      counters_;
  absl::flat_hash_map<std::string, std::unique_ptr<QueueBuffer<>>> gauges_;
  absl::flat_hash_map<std::string, std::unique_ptr<QueueBuffer<>>> histograms_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_BUFFERED_METRICS_EXPORTER_H_
