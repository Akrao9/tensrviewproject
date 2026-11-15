# syntax=docker/dockerfile:1
FROM rust:1.99.1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 libwayland-dev libxkbcommon-dev mesa-vulkan-drivers \
    vulkan-tools && rm -rf /var/lib/apt/lists/*
COPY --from=rune / /work/rune/
WORKDIR /work/tensrviewproject/experiments/moe-bakeoff
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src/ src/
# generate.py needs the SDK and torch, so the fixture is made on the host and passed in as a
# named build context (`--build-context seeded=artifacts/seeded`).
COPY --from=seeded seeded.evidence.json artifacts/stats.json
RUN rustup target add wasm32-unknown-unknown && cargo build --locked

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 libvulkan1 libxkbcommon0 mesa-vulkan-drivers \
    vulkan-tools && rm -rf /var/lib/apt/lists/*
COPY --from=build /work/tensrviewproject/experiments/moe-bakeoff/target/debug/moe-inspector /usr/local/bin/moe-inspector
ENV WGPU_BACKEND=vulkan RUNE_GPU_LOG=1 XDG_RUNTIME_DIR=/tmp
CMD ["moe-inspector", "--snap", "/output/rune-linux.png", "--tree", "/output/rune-linux-tree.json", "--theme", "dark", "--size", "1160x720", "--scale", "1"]
