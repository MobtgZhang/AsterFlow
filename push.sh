#!/usr/bin/env bash
# 位于仓库外：按主题拆分 git add / commit，最后 push。
# 用法：bash /home/mobtgzhang/github/push.sh
# 环境变量：ASTERFLOW_REPO 可覆盖仓库路径（默认 ~/github/AsterFlow）。
set -euo pipefail

REPO="${ASTERFLOW_REPO:-$HOME/github/AsterFlow}"
cd "$REPO"

commit_if_staged() {
  local msg="$1"
  if git diff --cached --quiet; then
    echo "[跳过] 无暂存变更: $msg"
  else
    git commit -m "$msg"
  fi
}

# --- 1. 文档 ---
git add docs/
commit_if_staged "docs: 设计说明、PyTorch 对照、构建与教程等"

# --- 2. aster_native（勿添加 aster_native/build/）---
git add aster_native/CMakeLists.txt aster_native/README.md
git add aster_native/include/ aster_native/tests/
git add aster_native/AsterNative/CMakeLists.txt
git add aster_native/AsterNative/ops/OpRegistry.h
git add aster_native/AsterNative/c_api/
git add aster_native/AsterNative/native/cpu/
commit_if_staged "feat(aster_native): C ABI、CPU Float32 内核、单测与 CMake"

# --- 3. CI ---
git add .github/
commit_if_staged "ci: 工作流（含 aster_native 构建）"

# --- 4. libasterflow ---
git add libasterflow/
commit_if_staged "feat(libasterflow): C 实现与头文件"

# --- 5. Julia 核心 ---
git add src/tensor.jl src/layout.jl src/device_tensor.jl \
  src/dispatch.jl src/accelerator_dispatch.jl src/native.jl \
  src/libasterflow.jl src/libasterflow_native.jl src/AsterFlow.jl \
  src/autograd.jl src/ops.jl src/ops_helpers.jl src/graph.jl \
  src/amp.jl src/checkpoint.jl src/grad_mode.jl src/devices.jl \
  src/loading.jl 2>/dev/null || true
git add src/distributed/ 2>/dev/null || true
git add src/storage.jl 2>/dev/null || true
commit_if_staged "feat(core): 张量、调度、autograd、可选 aster_native 对接"

# --- 6. compile ---
git add src/compile/
commit_if_staged "feat(compile): IR、trace、序列化、融合与 codegen"

# --- 7. nn / 数据 / 优化器 ---
git add src/layers/ src/losses/ src/transformer/ src/data/ src/optimise/
commit_if_staged "feat(nn): 层、损失、Transformer、DataLoader、优化器"

# --- 8. 扩展 ---
git add ext/
commit_if_staged "ext: CUDA / ROCm 等包扩展"

# --- 9. examples ---
git add examples/
commit_if_staged "examples: 轻量样例（MNIST 风格、语音占位、字符 LM）"

# --- 10. benchmark（目录存在则加入）---
if [ -d benchmark ]; then
  git add benchmark/
  commit_if_staged "bench: 基准或占位"
fi

# --- 11. 测试 ---
git add test/
commit_if_staged "test: runtests 与其它测试更新"

# --- 12. 剩余文件（请确认不含 build 产物）---
git status --short
set +e
git add -- . ':(exclude)aster_native/build'
git diff --cached --quiet
empty=$?
set -e
if [ "$empty" -ne 0 ]; then
  echo "以下将提交尚未归入上面对话的剩余变更（请检查 git diff --cached）："
  git diff --cached --stat
  read -r -p "确认提交剩余变更？(y/N) " ans
  if [[ "${ans:-N}" =~ ^[yY]$ ]]; then
    git commit -m "chore: 其余修改"
  else
    git reset HEAD
    echo "已取消剩余提交，请手动处理。"
  fi
else
  echo "无剩余未归类变更。"
fi

git push origin HEAD
