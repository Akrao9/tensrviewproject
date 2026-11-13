# syntax=docker/dockerfile:1
FROM rust:1.99.1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 libwayland-dev libxkbcommon-dev mesa-vulkan-drivers \
    python3 vulkan-tools && rm -rf /var/lib/apt/lists/*
COPY --from=rune / /work/rune/
WORKDIR /work/tensrviewproject/experiments/moe-bakeoff
COPY Cargo.toml Cargo.lock generate.py rust-toolchain.toml ./
COPY src/ src/
RUN rustup target add wasm32-unknown-unknown && python3 generate.py && cargo build --locked

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfontconfig1 libvulkan1 libxkbcommon0 mesa-vulkan-drivers \
    vulkan-tools && rm -rf /var/lib/apt/lists/*
COPY --from=build /work/tensrviewproject/experiments/moe-bakeoff/target/debug/moe-inspector /usr/local/bin/moe-inspector
ENV WGPU_BACKEND=vulkan RUNE_GPU_LOG=1 XDG_RUNTIME_DIR=/tmp
CMD ["moe-inspector", "--snap", "/output/rune-linux.png", "--tree", "/output/rune-linux-tree.json", "--theme", "dark", "--size", "1160x720", "--scale", "1"]
