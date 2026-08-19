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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_TEST_UTIL_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_TEST_UTIL_H_

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tpu_raiden::telemetry {

// Picks an available ephemeral loopback TCP port for test servers.
// Retries up to 10 times with SO_REUSEADDR to avoid port collisions under load.
inline int PickUnusedPort() {
  for (int attempt = 0; attempt < 10; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      continue;
    }

    int on = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
      close(fd);
      continue;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      close(fd);
      continue;
    }

    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      close(fd);
      continue;
    }

    int port = ntohs(addr.sin_port);
    close(fd);
    if (port > 0) {
      return port;
    }
  }
  return 0;
}

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_TEST_UTIL_H_
