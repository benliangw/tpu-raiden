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

#include "tpu_sync/common/version.h"

#include <gtest/gtest.h>

namespace tpu_sync {

namespace {

TEST(VersionTest, VersionConstantsAreValid) {
  EXPECT_GT(kCurrentVersion, 0);
  EXPECT_GT(kSupportedMinVersion, 0);
  EXPECT_LE(kSupportedMinVersion, kCurrentVersion);
}

TEST(VersionTest, IsVersionSupported) {
  EXPECT_TRUE(IsVersionSupported(kCurrentVersion));
  EXPECT_TRUE(IsVersionSupported(kSupportedMinVersion));
  EXPECT_TRUE(IsVersionSupported(kCurrentVersion + 1));
  EXPECT_FALSE(IsVersionSupported(0));
}

}  // namespace
}  // namespace tpu_sync
