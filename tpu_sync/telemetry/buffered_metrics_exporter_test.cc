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

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tpu_sync/telemetry/metrics_backend.h"

namespace tpu_raiden::telemetry {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// -----------------------------------------------------------------------------
// QueueBuffer Tests
// -----------------------------------------------------------------------------

TEST(QueueBufferTest, EmptyBufferExtractAndResetReturnsEmpty) {
  QueueBuffer<16> buffer;
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, SequentialPushAndExtractAndReset) {
  QueueBuffer<8> buffer;
  buffer.Push(1.5);
  buffer.Push(2.5);
  buffer.Push(3.5);

  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(1.5, 2.5, 3.5));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());

  // Verify buffer can be reused after reset
  buffer.Push(4.5);
  buffer.Push(5.5);
  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(4.5, 5.5));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, CapacityBoundaryAndOverflowTruncation) {
  QueueBuffer<4> buffer;
  for (int i = 1; i <= 10; ++i) {
    buffer.Push(static_cast<double>(i));
  }

  // Exactly the first 4 elements should be stored; the remaining 6 dropped.
  EXPECT_THAT(buffer.ExtractAndReset(), ElementsAre(1.0, 2.0, 3.0, 4.0));
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, FloatingPointBitwisePreservation) {
  QueueBuffer<8> buffer;
  const double kInf = std::numeric_limits<double>::infinity();
  const double kSubnormal = 1e-308;

  buffer.Push(0.0);
  buffer.Push(-0.0);
  buffer.Push(-123.456789);
  buffer.Push(kInf);
  buffer.Push(kSubnormal);

  std::vector<double> extracted = buffer.ExtractAndReset();
  ASSERT_EQ(extracted.size(), 5);
  EXPECT_EQ(extracted[0], 0.0);
  EXPECT_TRUE(std::signbit(extracted[1]));
  EXPECT_DOUBLE_EQ(extracted[2], -123.456789);
  EXPECT_EQ(extracted[3], kInf);
  EXPECT_DOUBLE_EQ(extracted[4], kSubnormal);
}

TEST(QueueBufferTest, ConcurrentMultiThreadedPush) {
  constexpr size_t kCapacity = 1000;
  QueueBuffer<kCapacity> buffer;

  constexpr int kNumThreads = 8;
  constexpr int kPushesPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&buffer, t] {
      for (int i = 0; i < kPushesPerThread; ++i) {
        buffer.Push(static_cast<double>(t * 1000 + i));
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::vector<double> extracted = buffer.ExtractAndReset();
  EXPECT_EQ(extracted.size(), kNumThreads * kPushesPerThread);
  EXPECT_THAT(buffer.ExtractAndReset(), IsEmpty());
}

TEST(QueueBufferTest, BufferResetAndReuseCycle) {
  QueueBuffer<100> buffer(/*max_capacity=*/100);
  for (int i = 0; i < 30; ++i) {
    buffer.Push(static_cast<double>(i));
  }
  std::vector<double> first_extract = buffer.ExtractAndReset();
  EXPECT_EQ(first_extract.size(), 30);

  // Subsequent push cycle after reset
  for (int i = 0; i < 40; ++i) {
    buffer.Push(static_cast<double>(i * 2));
  }
  std::vector<double> second_extract = buffer.ExtractAndReset();
  EXPECT_EQ(second_extract.size(), 40);
}

TEST(QueueBufferTest, ConcurrentPushAndExtract) {
  constexpr size_t kCapacity = 50000;
  QueueBuffer<kCapacity> buffer;

  constexpr int kNumThreads = 4;
  constexpr int kPushesPerThread = 5000;
  std::atomic<bool> stop_consumer{false};
  std::atomic<size_t> total_extracted{0};

  std::thread consumer([&] {
    while (!stop_consumer.load(std::memory_order_relaxed)) {
      std::vector<double> batch = buffer.ExtractAndReset();
      total_extracted.fetch_add(batch.size(), std::memory_order_relaxed);
      std::this_thread::yield();
    }
    std::vector<double> final_batch = buffer.ExtractAndReset();
    total_extracted.fetch_add(final_batch.size(), std::memory_order_relaxed);
  });

  std::vector<std::thread> producers;
  producers.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    producers.emplace_back([&buffer, t] {
      for (int i = 0; i < kPushesPerThread; ++i) {
        buffer.Push(static_cast<double>(t * 10000 + i));
      }
    });
  }

  for (auto& producer : producers) {
    producer.join();
  }
  stop_consumer.store(true, std::memory_order_relaxed);
  consumer.join();

  EXPECT_EQ(total_extracted.load(), kNumThreads * kPushesPerThread);
}

// -----------------------------------------------------------------------------
// LockFreeCounterAccumulator Tests
// -----------------------------------------------------------------------------

TEST(LockFreeCounterAccumulatorTest, InitialValueZero) {
  LockFreeCounterAccumulator counter;
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

TEST(LockFreeCounterAccumulatorTest, BasicAddAndReset) {
  LockFreeCounterAccumulator counter;
  counter.Add(100);
  counter.Add(50);
  EXPECT_EQ(counter.ExchangeAndReset(), 150);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);

  // Verify counter can be reused after reset
  counter.Add(25);
  EXPECT_EQ(counter.ExchangeAndReset(), 25);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

TEST(LockFreeCounterAccumulatorTest, ConcurrentAddStress) {
  LockFreeCounterAccumulator counter;
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 10000;
  constexpr uint64_t kDeltaPerIter = 5;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&counter] {
      for (int i = 0; i < kIterations; ++i) {
        counter.Add(kDeltaPerIter);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  constexpr uint64_t kExpectedTotal =
      static_cast<uint64_t>(kNumThreads) * kIterations * kDeltaPerIter;
  EXPECT_EQ(counter.ExchangeAndReset(), kExpectedTotal);
  EXPECT_EQ(counter.ExchangeAndReset(), 0);
}

// -----------------------------------------------------------------------------
// BufferedMetricsExporter Tests
// -----------------------------------------------------------------------------

TEST(BufferedMetricsExporterTest, EmptySamplesWhenNoMetricsUpdated) {
  BufferedMetricsExporter exporter;
  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest, GetTextSnapshotReturnsEmpty) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  EXPECT_EQ(exporter.GetTextSnapshot(), "");
}

TEST(BufferedMetricsExporterTest, CounterAccumulationAndReset) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 50);
  exporter.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 200);
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, {}, 3);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total"],
            (std::vector<double>{150.0}));
  EXPECT_EQ(samples["tpu_raiden_received_bytes_total"],
            (std::vector<double>{200.0}));
  EXPECT_EQ(samples["tpu_raiden_transfer_failures_total"],
            (std::vector<double>{3.0}));

  auto reset_samples = exporter.GetAndResetMetricSamples();
  EXPECT_THAT(reset_samples, IsEmpty());
}

TEST(BufferedMetricsExporterTest, GaugeAndHistogramMetricsExport) {
  MetricMetadata custom_metrics[] = {
      MetricMetadata{
          .name = "active_connections",
          .description = "Number of active connections.",
          .type = MetricType::kGauge,
      },
      MetricMetadata{
          .name = "transfer_latency_seconds",
          .description = "Transfer latency in seconds.",
          .type = MetricType::kHistogram,
      },
  };

  BufferedMetricsExporter exporter(custom_metrics);

  exporter.SetGauge("active_connections", {}, 12.0);
  exporter.SetGauge("active_connections", {}, 15.0);
  exporter.ObserveHistogram("transfer_latency_seconds", {}, 0.005);
  exporter.ObserveHistogram("transfer_latency_seconds", {}, 0.015);

  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_active_connections"],
            (std::vector<double>{12.0, 15.0}));
  EXPECT_EQ(samples["tpu_raiden_transfer_latency_seconds"],
            (std::vector<double>{0.005, 0.015}));

  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest, MetricLabelsPassedWithoutError) {
  BufferedMetricsExporter exporter;
  MetricLabel labels[] = {
      {"direction", "push"},
      {"peer_id", "42"},
  };

  exporter.IncrementCounter(metric_names::kSentBytesTotal, labels, 1024);
  auto samples = exporter.GetAndResetMetricSamples();
  EXPECT_EQ(samples["tpu_raiden_sent_bytes_total"],
            (std::vector<double>{1024.0}));
}

TEST(BufferedMetricsExporterTest, UnknownMetricNameIsIgnoredSafely) {
  BufferedMetricsExporter exporter;
  exporter.IncrementCounter("nonexistent_counter", {}, 100);
  exporter.SetGauge("nonexistent_gauge", {}, 42.0);
  exporter.ObserveHistogram("nonexistent_histogram", {}, 1.23);

  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

TEST(BufferedMetricsExporterTest,
     InterleavedConcurrentEmissionsAndExtractions) {
  BufferedMetricsExporter exporter;
  constexpr int kNumWriters = 6;
  constexpr int kIncrementsPerWriter = 2000;
  constexpr uint64_t kDelta = 1;

  std::atomic<bool> writers_done{false};
  std::atomic<double> total_extracted{0.0};

  // Reader thread performing continuous extractions concurrently with writers
  std::thread reader([&] {
    while (!writers_done.load(std::memory_order_relaxed)) {
      auto samples = exporter.GetAndResetMetricSamples();
      if (auto it = samples.find("tpu_raiden_sent_bytes_total");
          it != samples.end()) {
        for (double v : it->second) {
          total_extracted.fetch_add(v, std::memory_order_relaxed);
        }
      }
    }
    // Final drain after writers finish
    auto final_samples = exporter.GetAndResetMetricSamples();
    if (auto it = final_samples.find("tpu_raiden_sent_bytes_total");
        it != final_samples.end()) {
      for (double v : it->second) {
        total_extracted.fetch_add(v, std::memory_order_relaxed);
      }
    }
  });

  std::vector<std::thread> writers;
  writers.reserve(kNumWriters);
  for (int w = 0; w < kNumWriters; ++w) {
    writers.emplace_back([&] {
      for (int i = 0; i < kIncrementsPerWriter; ++i) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, kDelta);
      }
    });
  }

  for (auto& writer : writers) {
    writer.join();
  }
  writers_done.store(true, std::memory_order_relaxed);
  reader.join();

  constexpr double kExpectedTotal =
      static_cast<double>(kNumWriters * kIncrementsPerWriter * kDelta);
  EXPECT_DOUBLE_EQ(total_extracted.load(), kExpectedTotal);
  EXPECT_THAT(exporter.GetAndResetMetricSamples(), IsEmpty());
}

}  // namespace
}  // namespace tpu_raiden::telemetry
