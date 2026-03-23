/**
 * AsterFlow 原生张量运行时公共头。
 * 将 `src/` 加入包含路径后使用：`#include <AsterNative/AsterNative.h>`。
 */
#pragma once

#include <AsterNative/Backend.h>
#include <AsterNative/Device.h>
#include <AsterNative/Dispatch.h>
#include <AsterNative/Context.h>
#include <AsterNative/core/DispatchKey.h>
#include <AsterNative/core/Layout.h>
#include <AsterNative/core/ScalarType.h>
#include <AsterNative/core/Storage.h>
#include <AsterNative/core/TensorImpl.h>
#include <AsterNative/core/TensorOptions.h>
#include <AsterNative/detail/Exception.h>
#include <AsterNative/impl/DeviceGuard.h>
#include <AsterNative/accelerator/Accelerator.h>
#include <AsterNative/autograd/GradMode.h>
#include <AsterNative/ops/OpRegistry.h>
